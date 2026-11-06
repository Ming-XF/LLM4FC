#!/usr/bin/env python3
"""融合 token x 知识集 余弦相似度热图 (分析实验二)。

数据来源 (全部 CPU):
  - fusion 向量: ckpt 的 knowledge_fusion_tokens 快照 (3,16,4096),
    = mapper(LLM 词嵌入), 即训练时融合 token 的精确值;
  - 知识集: extract_bank_reproduce.py 复现保存的银行 token_ids
    (与 ckpt 逐位一致) 经 LLM 输入词嵌入矩阵索引得到的词向量 (171,4096);
  - 注意力排序: analysis/exp1/results/stats.json 的 mean_attn。
产物:
  - results/cosine_fc_psd.npz (fc/psd 余弦矩阵 16x171 + token_ids + 词面);
  - results/cosine_fc.png / cosine_psd.png — 16 行 x 全 171 列热图,
    行按注意力降序 (侧标 "t{idx} · {attn}"), 列每 10 个标一次词面。
运行: sh analysis/exp2/run_cosine.sh
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

CKPT = ("output_dir/output_dir_knowledge_s42_a3a/"
        "KnowledgeLLM_DiseaseCAUEEG_train")
BANK_PATH = "analysis/exp2/results/knowledge_bank_s42.npz"
LLM_PATH = "./model/deepseek-r1-distill-llama-8B"
LLM_SHARD = "model/deepseek-r1-distill-llama-8B/model-00001-of-000002.safetensors"
OUT_DIR = "analysis/exp2/results"
ASPECTS = ("fc", "psd")          # no_raw 主方法


def load_embedding_weight():
    from safetensors import safe_open
    with safe_open(LLM_SHARD, framework="pt", device="cpu") as f:
        return f.get_tensor("model.embed_tokens.weight").float()   # (V, 4096)


def decode_surface(tokenizer, token_id):
    tok = tokenizer.convert_ids_to_tokens(int(token_id))
    tok = tok.replace("Ġ", " ").replace("▁", " ")
    return tok.strip() or f"<{token_id}>"


def main():
    from transformers import AutoTokenizer

    os.makedirs(OUT_DIR, exist_ok=True)
    sd = torch.load(os.path.join(CKPT, "KnowledgeLLM-0.bin"),
                    map_location="cpu", weights_only=False)
    fusion = sd["knowledge_fusion_tokens"].float()                 # (3,16,4096)
    bank_npz = np.load(BANK_PATH, allow_pickle=False)
    stats = json.load(open("analysis/exp1/results/stats.json",
                           encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
    E = load_embedding_weight()

    saved = {}
    for i, aspect in enumerate(ASPECTS):
        bank_ids = torch.as_tensor(bank_npz[f"{aspect}_token_ids"])
        ckpt_ids = sd[f"knowledge_mappers.{i}.token_ids"]
        assert bool((bank_ids == ckpt_ids).all()), (
            f"{aspect}: 银行 token_ids 与 ckpt 不一致")
        emb = E[bank_ids]                                          # (171, 4096)
        cos = F.normalize(fusion[i], dim=-1) @ F.normalize(
            emb, dim=-1).T                                         # (16, 171)
        saved[aspect] = {
            "cosine": cos.numpy(),
            "token_ids": bank_ids.numpy(),
            "words": [decode_surface(tokenizer, t) for t in bank_ids],
        }
        attn = np.array(stats["caueeg"][aspect]["mean_attn"])      # (16,)
        order = np.argsort(-attn)
        print(f"[{aspect}] cosine min/mean/max = "
              f"{cos.min():.3f} / {cos.mean():.3f} / {cos.max():.3f}")
        for grp, idx in (("高注意力(前3)", order[:3]),
                         ("低注意力(后3)", order[-3:])):
            m = cos[idx].mean().item()
            print(f"   {grp} rows 平均余弦 = {m:.3f}")
        make_figure(aspect, saved[aspect], attn, order)

    np.savez(os.path.join(OUT_DIR, "cosine_fc_psd.npz"),
             fc_cosine=saved["fc"]["cosine"],
             psd_cosine=saved["psd"]["cosine"],
             fc_token_ids=saved["fc"]["token_ids"],
             psd_token_ids=saved["psd"]["token_ids"],
             fc_words=np.array(saved["fc"]["words"]),
             psd_words=np.array(saved["psd"]["words"]))
    print(f"结果已写入 {OUT_DIR}")


def make_figure(aspect, rec, attn, order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cos = rec["cosine"]                                            # (16,171)
    words = rec["words"]
    n_cols = cos.shape[1]
    fig, ax = plt.subplots(figsize=(18, 4.2))
    im = ax.imshow(cos[order], aspect="auto", cmap="viridis",
                   vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(16))
    ax.set_yticklabels(
        [f"t{idx} · {attn[idx]:.3f}" for idx in order], fontsize=9)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        [words[j] if j % 10 == 0 else "" for j in range(n_cols)],
        rotation=90, fontsize=6)
    ax.set_xlabel(f"{aspect.upper()} knowledge bank tokens (171, "
                  "every 10th labeled)")
    ax.set_ylabel("Fusion token (sorted by attention)")
    ax.set_title(f"{aspect.upper()} fusion token x knowledge bank cosine "
                 "similarity (s42 CAUEEG-trained)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02,
                 label="cosine similarity")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"cosine_{aspect}.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
