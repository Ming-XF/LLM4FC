#!/usr/bin/env python
"""KnowledgeLLM 融合 token 语义解码 (分析实验二)。

融合 token = KnowledgeMapper 对银行词嵌入的线性组合（mapping.weight: 16 x
银行token数，无偏置），因此每个融合 token 可由 mapper 权重解码为组成它的
银行词。直接从 ckpt 读取训练时的 token_ids 与 mapping.weight（绕开启动时
银行重抽取的 token 数不一致问题），用 LLM tokenizer 把银行 token id 还原为
词面，并与实验一的平均注意力交叉引用。

- 纯 CPU，不加载 8B 模型；
- 每个融合 token 报告 top-8 贡献银行词（按 |权重| 排序，含符号）；
- 输出：analysis/exp2/results/{fusion_decoding_fc,psd}.txt +
  mapping_weights.npz。

运行：sh analysis/exp2/run_decode.sh
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

CKPT = ("output_dir/output_dir_knowledge_s42_a3a/"
        "KnowledgeLLM_DiseaseCAUEEG_train")
LLM_PATH = "./model/deepseek-r1-distill-llama-8B"
OUT_DIR = "analysis/exp2/results"
TOP_K = 8
ACTIVE_ASPECTS = ("fc", "psd")          # no_raw 主方法


def load_ckpt_weights(path):
    sd = torch.load(os.path.join(path, "KnowledgeLLM-0.bin"),
                    map_location="cpu", weights_only=False)
    out = {}
    for i, aspect in enumerate(("fc", "psd", "raw")):
        out[aspect] = {
            "token_ids": sd[f"knowledge_mappers.{i}.token_ids"].numpy(),
            "mapping": sd[f"knowledge_mappers.{i}.mapping.weight"].numpy(),
        }
    return out


def load_exp1_attention():
    stats = json.load(open("analysis/exp1/results/stats.json", encoding="utf-8"))
    return {a: stats["caueeg"][a]["mean_attn"] for a in ("fc", "psd")}


def decode_surface(tokenizer, token_id):
    tok = tokenizer.convert_ids_to_tokens(int(token_id))
    tok = tok.replace("Ġ", " ").replace("▁", " ")
    return tok.strip() or f"<{token_id}>"


def main():
    from transformers import AutoTokenizer
    os.makedirs(OUT_DIR, exist_ok=True)

    banks = load_ckpt_weights(CKPT)
    attn_mean = load_exp1_attention()
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)

    saved = {}
    for aspect in ACTIVE_ASPECTS:
        ids = banks[aspect]["token_ids"]
        W = banks[aspect]["mapping"]              # (16, n_bank)
        n_fusion, n_bank = W.shape
        lines = []
        lines.append(f"== {aspect.upper()} 融合 token 语义解码 ==")
        lines.append(f"银行 token 数: {n_bank}，融合 token 数: {n_fusion}")
        lines.append("")
        # 融合 token 按实验一平均注意力降序排列，先看"最忙"的 token
        order = np.argsort(-np.array(attn_mean[aspect]))
        for rank, j in enumerate(order):
            w = W[j]
            idx = np.argsort(-np.abs(w))[:TOP_K]
            words = []
            for k in idx:
                surf = decode_surface(tokenizer, ids[k])
                words.append((surf, float(w[k])))
            attn = attn_mean[aspect][j]
            lines.append(
                f"[{rank + 1}/{n_fusion}] 融合 token #{j} "
                f"(平均注意力 {attn:.4f})")
            for pos, (surf, weight) in enumerate(words, 1):
                lines.append(
                    f"    {pos:>2}. {surf:<24} {'+' if weight >= 0 else '-'}"
                    f"{abs(weight):.4f}")
            lines.append("")
        text = "\n".join(lines)
        out_path = os.path.join(OUT_DIR, f"fusion_decoding_{aspect}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(text)
        saved[aspect] = {"token_ids": ids, "mapping": W}

    np.savez(os.path.join(OUT_DIR, "mapping_weights.npz"),
             fc_token_ids=saved["fc"]["token_ids"],
             fc_mapping=saved["fc"]["mapping"],
             psd_token_ids=saved["psd"]["token_ids"],
             psd_mapping=saved["psd"]["mapping"])
    print(f"\n结果已写入 {OUT_DIR}")


if __name__ == "__main__":
    main()
