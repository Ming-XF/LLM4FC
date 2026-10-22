"""Two-level semantic prototype banks for the three EEG knowledge families.

The frozen language vocabulary is used only as a deterministic semantic
candidate source. A first candidate pool is linearly fused into several
aspect-specific anchors; each anchor then owns a second description pool
which is compressed into the prototypes consumed by reprogramming.
"""

import torch
import torch.nn as nn


_EPS = 1e-8


def _group_average_init(weight):
    """Initialize each output as an average over a sorted candidate group."""
    outputs, inputs = weight.shape
    with torch.no_grad():
        weight.zero_()
        for i in range(outputs):
            start = (i * inputs) // outputs
            end = ((i + 1) * inputs) // outputs
            end = max(end, start + 1)
            weight[i, start:end] = 1.0 / float(end - start)


def _row_normalize(x):
    return x / x.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(_EPS)


class SemanticPrototypeBank(nn.Module):
    """Build shared two-level semantic banks for three aspect queries.

    Shapes:
      first candidates:       (A, K, d_llm)
      refined anchors:        (A, N, d_llm)
      descriptions:           (A, N, M, d_llm)
      output prototypes:      (A, N, P, d_llm)

    Candidate ids are fixed after initialization. The two Linear layers use
    the existing signed candidate-space mapping style and remain trainable.
    """

    def __init__(self, phrase_bank, word_embeddings, tokenizer,
                 anchor_candidates=1000, num_anchors=10,
                 description_candidates=1000, num_prototypes=100,
                 similarity="centered_cosine", aspect_mix=0.6,
                 anchor_fusion_init=0.1):
        super().__init__()
        num_aspects, d_llm = phrase_bank.shape
        vocab_size, emb_dim = word_embeddings.shape
        if emb_dim != d_llm:
            raise ValueError("词表 embedding 维度与短语锚不一致")
        if anchor_candidates <= 0 or num_anchors <= 0:
            raise ValueError("anchor_candidates 和 num_anchors 必须为正")
        if description_candidates <= 0 or num_prototypes <= 0:
            raise ValueError(
                "description_candidates 和 num_prototypes 必须为正")
        if num_anchors > anchor_candidates:
            raise ValueError("num_anchors 不能大于 anchor_candidates")
        if num_prototypes > description_candidates:
            raise ValueError(
                "num_prototypes 不能大于 description_candidates")
        if similarity not in {"cosine", "centered_cosine"}:
            raise ValueError(
                "similarity 必须是 cosine 或 centered_cosine")
        if not 0.0 <= aspect_mix <= 1.0:
            raise ValueError("aspect_mix 必须位于 [0, 1]")
        if anchor_fusion_init < 0:
            raise ValueError("anchor_fusion_init 不能为负")

        self.num_aspects = num_aspects
        self.anchor_candidates = anchor_candidates
        self.num_anchors = num_anchors
        self.description_candidates = description_candidates
        self.num_prototypes = num_prototypes
        self.similarity = similarity
        self.aspect_mix = aspect_mix
        self.anchor_fusion = nn.Parameter(
            torch.tensor(float(anchor_fusion_init)))

        # Candidate retrieval is done once in fp32 so the semantic bank stays
        # deterministic and domain-independent during training.
        # Keep a low-precision copy for the persistent banks. Similarity
        # retrieval is still performed in fp32 below for stable top-k ids.
        embedding_storage_dtype = word_embeddings.dtype
        emb = word_embeddings.detach().float()
        if similarity == "centered_cosine":
            center = emb.mean(dim=0, keepdim=True)
            emb_for_sim = emb - center
            phrase_for_sim = phrase_bank.detach().float() - center
        else:
            center = None
            emb_for_sim = emb
            phrase_for_sim = phrase_bank.detach().float()
        emb_n = _row_normalize(emb_for_sim)
        phrase_n = _row_normalize(phrase_for_sim)

        special_ids = sorted({
            int(i) for i in tokenizer.all_special_ids
            if i is not None and 0 <= int(i) < vocab_size
        })
        if len(special_ids) >= vocab_size:
            raise ValueError("词表中没有可用于语义候选的普通 Token")
        special = torch.tensor(
            special_ids, dtype=torch.long, device=emb.device)

        first_scores = phrase_n @ emb_n.T
        if special.numel():
            first_scores[:, special] = -float("inf")
        valid_count = vocab_size - special.numel()
        if anchor_candidates > valid_count:
            raise ValueError(
                f"anchor_candidates={anchor_candidates} 超过可用词表 Token 数 "
                f"{valid_count}")
        first_ids = first_scores.topk(anchor_candidates, dim=-1).indices
        first_embeddings = emb[first_ids]
        self.register_buffer("anchor_candidate_ids", first_ids.long())
        self.register_buffer(
            "anchor_candidate_embeddings",
            first_embeddings.to(dtype=embedding_storage_dtype),
            persistent=False)

        # First mapping: K candidates -> N refined knowledge anchors.
        self.anchor_mapper = nn.Linear(
            anchor_candidates, num_anchors, bias=False)
        _group_average_init(self.anchor_mapper.weight)
        with torch.no_grad():
            refined_init = torch.einsum(
                "nk,akd->and", self.anchor_mapper.weight.float(),
                first_embeddings)

        # The second-level ids are fixed after initialization. The mapping
        # itself is trainable, but retrieval does not jump during training.
        refined_for_sim = refined_init
        if similarity == "centered_cosine":
            refined_for_sim = refined_for_sim - center
        refined_n = _row_normalize(refined_for_sim)
        joint_query = (
            aspect_mix * phrase_n.unsqueeze(1)
            + (1.0 - aspect_mix) * refined_n
        )
        joint_query = _row_normalize(joint_query)
        second_scores = torch.einsum(
            "and,vd->anv", joint_query, emb_n)
        if special.numel():
            second_scores[:, :, special] = -float("inf")
        if description_candidates > valid_count:
            raise ValueError(
                f"description_candidates={description_candidates} 超过可用词表 "
                f"Token 数 {valid_count}")
        description_ids = second_scores.topk(
            description_candidates, dim=-1).indices
        description_embeddings = emb[description_ids]
        self.register_buffer("description_ids", description_ids.long())
        self.register_buffer(
            "description_embeddings",
            description_embeddings.to(dtype=embedding_storage_dtype),
            persistent=False)

        # Second mapping: M descriptions -> P description prototypes.
        self.description_mapper = nn.Linear(
            description_candidates, num_prototypes, bias=False)
        _group_average_init(self.description_mapper.weight)

    def refined_anchors(self):
        """Return trainable refined anchors with shape ``(A, N, d_llm)``."""
        return torch.einsum(
            "nk,akd->and", self.anchor_mapper.weight,
            self.anchor_candidate_embeddings.to(
                dtype=self.anchor_mapper.weight.dtype))

    def forward(self, aspect_ids=None):
        """Return description prototypes as ``(A, N, P, d_llm)``."""
        prototypes = torch.einsum(
            "pm,anmd->anpd", self.description_mapper.weight,
            self.description_embeddings.to(
                dtype=self.description_mapper.weight.dtype))
        # Keep the first-level fused knowledge trainable in the forward path;
        # the frozen second-level ids only determine its semantic neighborhood.
        prototypes = prototypes + self.anchor_fusion * self.refined_anchors().unsqueeze(2)
        if aspect_ids is not None:
            prototypes = prototypes[aspect_ids]
        return prototypes
