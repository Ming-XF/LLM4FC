"""
Graph-Conditioned LoRA (GC-LoRA) for TimeLLM.

Architecture
------------
Standard LoRA wraps a frozen Linear with a low-rank bypass::

    output = W·x  +  (alpha/r) · x·A·B       (A: d_in→r,  B: r→d_out)

GC-LoRA inserts a graph-convolution step between A and B so that every
node-token can aggregate information from its functional neighbours
*before* the LoRA B projection::

    h     = A(x)                             (B, T, C, r)
    h_agg = FC_adj · h                       intra-window GCN along C
    Δ     = B(h_agg)                         (B, T, C, d_out)

Because the GCN sits in the low-rank bottleneck (dimension r), it adds
negligible FLOPs while allowing cross-node information injection that
standard LoRA cannot provide.

When ``use_graph_cond=False`` the module falls back to standard LoRA so
the same code path supports ablation experiments.
"""

from __future__ import annotations

import logging
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# GC-LoRA Linear wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class GCLoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen ``nn.Linear`` with optional
    graph-conditioned neighbour aggregation between A and B.

    The *base* linear is kept frozen (its weights never receive gradients).
    Two small trainable matrices ``lora_A`` and ``lora_B`` form the low-rank
    bypass.  When ``use_graph_cond=True``, the intermediate representation is
    reshaped to ``(T, C, r)`` and passed through a one-hop GCN along the
    channel dimension using the functional-connectivity adjacency stored in
    ``self.fc_adj``.

    Parameters
    ----------
    base_linear : nn.Linear
        The original (frozen) linear layer.
    rank : int
        LoRA rank *r*.
    alpha : float
        Scaling factor; final multiplier = ``alpha / rank``.
    dropout : float
        Dropout probability applied after ``lora_A``.
    num_nodes : int
        Number of EEG channels *C* (typically 19).
    num_windows : int
        Number of time windows *T* (typically 10).
    use_graph_cond : bool
        If True, apply GCN aggregation between A and B (GC-LoRA).
        If False, behave as standard LoRA.

    Runtime context (set by ``Model.forward`` before calling the LLM)
    -----------------------------------------------------------------
    ``fc_adj`` : torch.Tensor | None
        ``(B, T, C, C)`` — per-window FC adjacency.  Set before the LLM
        forward and cleared afterward.
    ``prompt_len`` : int
        Number of prompt tokens *P_skip* prepended before the patch tokens.
        Defaults to 0 (all tokens are patches).
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        num_nodes: int = 19,
        num_windows: int = 10,
        use_graph_cond: bool = False,
    ):
        super().__init__()
        d_in = base_linear.in_features
        d_out = base_linear.out_features

        # ── Frozen base ──
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad = False

        # ── Trainable LoRA matrices ──
        self.lora_A = nn.Linear(d_in, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.num_nodes = num_nodes
        self.num_windows = num_windows
        self.use_graph_cond = use_graph_cond

        # Runtime context — populated by Model.forward before each LLM call
        self.fc_adj: torch.Tensor | None = None   # (B, T, C, C)
        self.prompt_len: int = 0                   # P_skip

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for A, zeros for B (so Δ = 0 at start)."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, S, d_in)`` — full sequence (prompt + patches).

        Returns:
            ``(B, S, d_out)`` — base output + LoRA delta.
        """
        # ── Frozen base forward ──
        result = self.base_linear(x)

        P = self.prompt_len
        if P == 0:
            return result  # no patch context — skip LoRA

        B, S, d_in = x.shape
        T, C = self.num_windows, self.num_nodes
        r = self.rank

        if P >= S:
            return result  # sequence is all prompt, nothing to adapt

        # ── Slice patch tokens ──
        x_patches = x[:, P:, :]                          # (B, T*C, d_in)

        # ── Reshape to (B, T, C, d_in) ──
        # Current token order is time-first:
        #   C0T0, C1T0, ..., C18T0, C0T1, ..., C18T9
        # So consecutive groups of C tokens belong to the same time window.
        x_2d = x_patches.reshape(B, T, C, d_in)          # (B, T, C, d_in)

        # ── Cast to LoRA weight dtype ──
        # LLM hidden states are bfloat16, but DeepSpeed ZeRO-2 manages
        # trainable params in fp32.  Cast input to match lora_A's dtype
        # so the matmul succeeds, then cast delta back to the input dtype.
        lora_dtype = self.lora_A.weight.dtype
        x_2d = x_2d.to(dtype=lora_dtype)

        # ── LoRA A: d_in → r ──
        h = self.lora_A(self.dropout(x_2d))              # (B, T, C, r)

        # ── [GC-LoRA] GCN one-hop aggregation along C ──
        if self.use_graph_cond and self.fc_adj is not None:
            h_flat = h.reshape(B * T, C, r)              # (B*T, C, r)
            if self.fc_adj.dim() == 4:                    # (B, T, C, C)
                adj_flat = self.fc_adj.reshape(B * T, C, C)
            else:
                adj_flat = self.fc_adj                   # already (B*T, C, C)
            adj_flat = adj_flat.to(dtype=h_flat.dtype)
            h_agg = torch.bmm(adj_flat, h_flat)          # (B*T, C, r)
            h = h_agg.reshape(B, T, C, r)               # (B, T, C, r)
        # (standard LoRA: h passes through unchanged)

        # ── LoRA B: r → d_out ──
        delta = self.lora_B(h)                           # (B, T, C, d_out)

        # ── Cast delta back to input dtype ──
        delta = delta.to(dtype=result.dtype)

        # ── Flatten back to sequence ──
        delta_flat = delta.reshape(B, T * C, self.base_linear.out_features)
        delta_full = torch.cat([
            torch.zeros(B, P, delta_flat.size(-1),
                       device=x.device, dtype=delta_flat.dtype),
            delta_flat,
        ], dim=1)                                        # (B, S, d_out)

        return result + self.scaling * delta_full


# ═══════════════════════════════════════════════════════════════════════════════
# Injection utilities
# ═══════════════════════════════════════════════════════════════════════════════

def inject_lora_to_llm(
    transformer: nn.Module,
    llm_type: str,
    target_modules: list[str],
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.1,
    num_nodes: int = 19,
    num_windows: int = 10,
    use_graph_cond: bool = False,
) -> int:
    """Replace selected Linear layers inside the LLM decoder stack with
    :class:`GCLoRALinear` wrappers.

    Parameters
    ----------
    transformer:
        The LLM backbone — ``ChatGLMModel`` or ``LlamaModel`` (the object
        stored in ``self._transformer`` / ``self.llm``).
    llm_type:
        ``"chatglm"`` or ``"llama"``.
    target_modules:
        Short attribute names, e.g. ``["q_proj", "v_proj"]``.  For LLaMA
        these live under ``layer.self_attn.<name>`` or ``layer.mlp.<name>``;
        for ChatGLM they live under ``layer.attention.<name>`` or
        ``layer.mlp.<name>``.
    rank:
        LoRA rank *r*.
    alpha:
        LoRA scaling factor.
    dropout:
        Dropout probability.
    num_nodes:
        Number of EEG channels *C*.
    num_windows:
        Number of time windows *T*.
    use_graph_cond:
        Enable GC-LoRA (GCN between A and B).

    Returns
    -------
    n_injected : int
        Number of Linear layers that were replaced.
    """
    # ── Resolve the list of decoder layers ──
    if llm_type == 'chatglm':
        layers = transformer.layers
    elif llm_type == 'llama':
        layers = transformer.layers
    else:
        raise ValueError(f"Unsupported llm_type: {llm_type}")

    # ── Sub-module search paths per architecture ──
    if llm_type == 'chatglm':
        search_paths = ['attention', 'mlp']
    else:  # llama
        search_paths = ['self_attn', 'mlp']

    n_injected = 0
    na = 'GC-' if use_graph_cond else ''

    for layer_idx, layer in enumerate(layers):
        for path in search_paths:
            container = getattr(layer, path, None)
            if container is None:
                continue
            for target in target_modules:
                if not hasattr(container, target):
                    continue
                orig = getattr(container, target)
                if not isinstance(orig, nn.Linear):
                    continue

                wrapped = GCLoRALinear(
                    base_linear=orig,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    num_nodes=num_nodes,
                    num_windows=num_windows,
                    use_graph_cond=use_graph_cond,
                )
                setattr(container, target, wrapped)
                n_injected += 1
                _log.debug(
                    "layer[%d].%s.%s → %sGCLoRALinear(r=%d, α=%.1f)",
                    layer_idx, path, target, na, rank, alpha,
                )

    return n_injected


def set_gc_lora_context(
    model: nn.Module,
    fc_adj: torch.Tensor | None,
    prompt_len: int,
):
    """Walk all :class:`GCLoRALinear` sub-modules in *model* and set runtime
    graph context.  Call before the LLM forward.

    Parameters
    ----------
    model:
        The top-level ``Model`` (or any ``nn.Module`` containing
        ``GCLoRALinear`` children).
    fc_adj:
        ``(B, T, C, C)`` per-window FC adjacency, or ``None`` to clear.
    prompt_len:
        Number of prompt tokens *P_skip*.
    """
    for module in model.modules():
        if isinstance(module, GCLoRALinear):
            module.fc_adj = fc_adj
            module.prompt_len = prompt_len


def clear_gc_lora_context(model: nn.Module):
    """Clear graph context on all :class:`GCLoRALinear` sub-modules.
    Call after the LLM forward (safe cleanup)."""
    set_gc_lora_context(model, fc_adj=None, prompt_len=0)
