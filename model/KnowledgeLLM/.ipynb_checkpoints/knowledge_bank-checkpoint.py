"""Knowledge-bank extraction and loading utilities for KnowledgeLLM.

The bank is built offline from a frozen causal language model.  A bank entry
contains the vocabulary ids selected from the last ``K`` valid text positions
and a non-negative normalized weight for each unique id.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


ASPECTS = ("fc", "psd", "raw")

DEFAULT_KNOWLEDGE_TEXTS = {
    "fc": (
        "In EEG functional connectivity research, neurodegenerative and "
        "neurological disorders can be associated with altered connectivity "
        "strength, network topology, and interactions between brain regions."
    ),
    "psd": (
        "Reported EEG spectral findings in neurological disorders include "
        "changes in delta, theta, alpha, and beta band power and in the "
        "relative distribution of power across frequency bands."
    ),
    "raw": (
        "Abnormal EEG waveforms can include rhythmic slowing, sharp waves, "
        "spikes, spike-and-wave complexes, and other transient morphological "
        "patterns described in clinical neurophysiology."
    ),
}


def _as_1d(value, dtype):
    array = np.asarray(value)
    if array.ndim != 1:
        array = array.reshape(-1)
    return torch.as_tensor(array, dtype=dtype)


def _normalise_weights(weights, size):
    if weights is None:
        return torch.full((size,), 1.0 / max(size, 1), dtype=torch.float32)
    weights = _as_1d(weights, torch.float32)
    if weights.numel() != size:
        raise ValueError(
            f"Knowledge weights have {weights.numel()} values for {size} ids")
    if not torch.isfinite(weights).all():
        raise ValueError("Knowledge weights contain NaN or infinite values")
    # Accept either already-normalized probabilities or raw logit sums.  Raw
    # logits can be negative, so convert those to a stable positive prior.
    if bool((weights >= 0).all()) and float(weights.sum()) > 0:
        weights = weights / weights.sum()
    else:
        weights = torch.softmax(weights.float(), dim=0)
    return weights


def _entry(token_ids, weights=None):
    token_ids = _as_1d(token_ids, torch.long)
    if token_ids.numel() == 0:
        raise ValueError("Each KnowledgeLLM aspect needs at least one token id")
    if bool((token_ids < 0).any()):
        raise ValueError("Knowledge token ids must be non-negative")
    if weights is None:
        raw_weights = None
    else:
        raw_weights = _as_1d(weights, torch.float32)
        if raw_weights.numel() != token_ids.numel():
            raise ValueError(
                f"Knowledge weights have {raw_weights.numel()} values for "
                f"{token_ids.numel()} ids")
    unique_ids, inverse = torch.unique(token_ids, sorted=False,
                                       return_inverse=True)
    if raw_weights is None:
        merged_weights = None
    else:
        merged_weights = torch.zeros(unique_ids.numel(), dtype=torch.float32)
        merged_weights.scatter_add_(0, inverse, raw_weights)
    return {
        "token_ids": unique_ids,
        "weights": _normalise_weights(merged_weights, unique_ids.numel()),
    }


def _find_field(container, names):
    for name in names:
        if name in container:
            return container[name]
    return None


def load_knowledge_bank(path):
    """Load a bank from ``.npz``, ``.json`` or ``.pt``.

    Supported field names include ``fc_token_ids``/``fc_weights`` and the
    equivalent ``token_ids_fc``/``weights_fc`` form.  JSON and torch files may
    instead contain ``{"fc": {"token_ids": ..., "weights": ...}}``.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Knowledge bank does not exist: {path}")

    loaded = {}
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            for aspect in ASPECTS:
                ids = _find_field(
                    data,
                    (f"{aspect}_token_ids", f"token_ids_{aspect}",
                     f"{aspect}_ids"),
                )
                weights = _find_field(
                    data,
                    (f"{aspect}_raw_logit_sum", f"raw_logit_sum_{aspect}",
                     f"{aspect}_weights", f"weights_{aspect}",
                     f"{aspect}_logit_weights", f"{aspect}_logits"),
                )
                if ids is None:
                    raise ValueError(
                        f"Knowledge bank {path} is missing token ids for {aspect}")
                loaded[aspect] = _entry(ids, weights)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for aspect in ASPECTS:
            item = data.get(aspect, {})
            ids = item.get("token_ids", item.get("ids"))
            if ids is None:
                raise ValueError(
                    f"Knowledge bank {path} is missing token ids for {aspect}")
            loaded[aspect] = _entry(
                ids,
                item.get("raw_logit_sum",
                         item.get("weights", item.get("logits"))),
            )
    elif path.suffix.lower() in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
        for aspect in ASPECTS:
            item = data.get(aspect, {})
            ids = item.get("token_ids", item.get("ids"))
            if ids is None:
                raise ValueError(
                    f"Knowledge bank {path} is missing token ids for {aspect}")
            loaded[aspect] = _entry(
                ids,
                item.get("raw_logit_sum",
                         item.get("weights", item.get("logits"))),
            )
    else:
        raise ValueError("Knowledge bank must be .npz, .json, .pt or .pth")
    return loaded


def save_knowledge_bank(bank, path):
    """Persist an extracted bank for reuse during model training."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        fields = {}
        for aspect in ASPECTS:
            item = bank[aspect]
            fields[f"{aspect}_token_ids"] = item["token_ids"].cpu().numpy()
            fields[f"{aspect}_weights"] = item["weights"].cpu().numpy()
            if "raw_logit_sum" in item:
                fields[f"{aspect}_raw_logit_sum"] = item[
                    "raw_logit_sum"].cpu().numpy()
        np.savez_compressed(path, **fields)
    elif path.suffix.lower() == ".json":
        serializable = {}
        for aspect in ASPECTS:
            item = bank[aspect]
            serializable[aspect] = {
                "token_ids": item["token_ids"].cpu().tolist(),
                "weights": item["weights"].cpu().tolist(),
            }
            if "raw_logit_sum" in item:
                serializable[aspect]["raw_logit_sum"] = item[
                    "raw_logit_sum"].cpu().tolist()
        with path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)
    elif path.suffix.lower() in {".pt", ".pth"}:
        torch.save(bank, path)
    else:
        raise ValueError("Knowledge bank must be .npz, .json, .pt or .pth")


@torch.no_grad()
def extract_knowledge_bank(causal_lm, tokenizer, texts=None, top_k_positions=8,
                           top_m_tokens=32, temperature=1.0):
    """Extract three aspect-specific banks from a frozen causal LLM.

    The causal model must expose ``model`` (the transformer backbone) and
    ``lm_head``.  Only the final hidden layer is used; all returned tensors are
    CPU tensors and can be serialized for later training.
    """
    if top_k_positions <= 0 or top_m_tokens <= 0:
        raise ValueError("top_k_positions and top_m_tokens must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    texts = DEFAULT_KNOWLEDGE_TEXTS if texts is None else texts
    device = next(causal_lm.parameters()).device
    special_ids = set(int(i) for i in tokenizer.all_special_ids if i is not None)
    result = {}
    causal_lm.eval()

    for aspect in ASPECTS:
        text = texts[aspect]
        encoded = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        outputs = causal_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        logits = causal_lm.lm_head(hidden).float()[0]
        if attention_mask is not None:
            hidden_positions = attention_mask[0].nonzero(as_tuple=False).flatten()
        else:
            hidden_positions = torch.arange(input_ids.shape[1], device=device)
        # Keep the last K text positions, while excluding special positions
        # from candidate selection rather than from hidden-state indexing.
        positions = [
            int(pos) for pos in hidden_positions.tolist()
            if int(input_ids[0, pos]) not in special_ids
        ]
        if not positions:
            positions = hidden_positions.tolist()
        positions = positions[-top_k_positions:]

        vocab_size = logits.shape[-1]
        candidate_logits = logits[positions].clone()
        if special_ids:
            special = [i for i in special_ids if i < vocab_size]
            candidate_logits[:, special] = -float("inf")
        m = min(top_m_tokens, vocab_size - len(special_ids))
        top_values, top_ids = candidate_logits.topk(m, dim=-1)
        raw_sums = {}
        probability_sums = {}
        for row_values, row_ids in zip(top_values, top_ids):
            probs = torch.softmax(row_values / temperature, dim=-1)
            for value, probability, token_id in zip(
                    row_values.tolist(), probs.tolist(), row_ids.tolist()):
                token_id = int(token_id)
                raw_sums[token_id] = raw_sums.get(token_id, 0.0) + float(value)
                probability_sums[token_id] = (
                    probability_sums.get(token_id, 0.0) + float(probability))
        ids = sorted(probability_sums, key=probability_sums.get, reverse=True)
        result[aspect] = _entry(
            ids,
            [raw_sums[token_id] for token_id in ids],
        )
        result[aspect]["raw_logit_sum"] = torch.tensor(
            [raw_sums[token_id] for token_id in ids], dtype=torch.float32)
    return result
