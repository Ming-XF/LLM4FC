#!/usr/bin/env python
"""One-time knowledge-bank generation for the A1 ablation.

Extracts the pretrained bank with the finalized KnowledgeLLM parameters and
mirrors it with a random bank: per aspect, the same number of unique tokens,
sampled without replacement from the LLM vocabulary (special ids excluded),
uniform weights, fixed RNG seed.  Both files are written to --out-dir:

  pretrained_bank.npz  — authoritative pretrained bank (also reused by A2a)
  random_bank.npz      — A1 control bank, identical across training seeds

Run from the repo root:  python scripts/gen_random_bank.py
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.KnowledgeLLM.knowledge_bank import (
    ASPECTS,
    extract_knowledge_bank,
    save_knowledge_bank,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-path", default="./model/deepseek-r1-distill-llama-8B")
    parser.add_argument("--out-dir", default="output_dir/knowledge_banks")
    parser.add_argument("--top-k", type=int, default=8,
                        help="last hidden positions used for extraction")
    parser.add_argument("--top-m", type=int, default=32,
                        help="top vocabulary tokens per position")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for the random-bank token sampling")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_path, trust_remote_code=True)
    causal = AutoModelForCausalLM.from_pretrained(
        args.llm_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    if torch.cuda.is_available():
        causal = causal.cuda()

    pretrained = extract_knowledge_bank(
        causal, tokenizer,
        top_k_positions=args.top_k, top_m_tokens=args.top_m,
        temperature=args.temperature)

    vocab_size = int(causal.model.embed_tokens.weight.shape[0])
    special_ids = {int(i) for i in tokenizer.all_special_ids if i is not None}
    candidates = [i for i in range(vocab_size) if i not in special_ids]
    rng = random.Random(args.seed)

    random_bank = {}
    for aspect in ASPECTS:
        n = int(pretrained[aspect]["token_ids"].numel())
        if n > len(candidates):
            raise ValueError(
                f"Bank {aspect} needs {n} tokens but only "
                f"{len(candidates)} non-special candidates exist")
        ids = rng.sample(candidates, n)
        random_bank[aspect] = {
            "token_ids": torch.as_tensor(ids, dtype=torch.long),
            "weights": torch.full((n,), 1.0 / max(n, 1), dtype=torch.float32),
        }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_knowledge_bank(pretrained, out_dir / "pretrained_bank.npz")
    save_knowledge_bank(random_bank, out_dir / "random_bank.npz")

    print(f"vocab_size={vocab_size}, special_ids={len(special_ids)}, "
          f"seed={args.seed}", flush=True)
    for aspect in ASPECTS:
        print(f"  {aspect}: {pretrained[aspect]['token_ids'].numel()} tokens",
              flush=True)
    print(f"saved {out_dir / 'pretrained_bank.npz'}", flush=True)
    print(f"saved {out_dir / 'random_bank.npz'}", flush=True)


if __name__ == "__main__":
    main()
