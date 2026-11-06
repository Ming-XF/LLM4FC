#!/usr/bin/env python3
"""融合 token 透镜: 知识集归属与判定表 (实验二·部分 C)。

基于 exp2 的余弦矩阵 (16 x 171, 融合 token x 银行词), 纯 CPU 加工:
  - 归属: 每个银行词 argmax 余弦 -> 归属到某个融合 token;
  - 每融合 token 一行: 注意力 (exp1, CAUEEG 域) + 归属词数 +
    归属词全表 (按余弦降序, 内容判定的依据);
  - 按注意力降序排列;
  - 图: 两模态各一面板, 注意力横条 + top-3 归属词 + 归属词数。
内容性质判定不写死在代码里 (避免隐藏词典), 由人读表复核后登记。

产物: results/fusion_lens_profile.txt + results/fusion_lens_profile.png
运行: sh analysis/exp2/run_profile.sh
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

OUT_DIR = "analysis/exp2/results"
ASPECTS = ("fc", "psd")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cos_npz = np.load("analysis/exp2/results/cosine_fc_psd.npz",
                      allow_pickle=True)
    stats = json.load(open("analysis/exp1/results/stats.json",
                           encoding="utf-8"))

    lines = []
    lines.append("融合 token 透镜: 知识集归属与判定表 (s42 a3a)")
    lines.append("口径: 余弦矩阵来自 exp2 (16 融合 token x 171 银行词); "
                 "归属 = 每银行词 argmax 余弦; 注意力 = exp1 CAUEEG 域; "
                 "按注意力降序。\n")
    for aspect in ASPECTS:
        cos = cos_npz[f"{aspect}_cosine"]          # (16, 171)
        words = [str(w) for w in cos_npz[f"{aspect}_words"]]
        attn = np.array(stats["caueeg"][aspect]["mean_attn"])
        owner = cos.argmax(axis=0)                 # (171,) -> 融合 token idx
        order = np.argsort(-attn)

        lines.append(f"===== {aspect.upper()} =====")
        for rank, j in enumerate(order):
            owned_idx = np.where(owner == j)[0]
            owned_idx = owned_idx[np.argsort(-cos[j, owned_idx])]
            owned_words = [words[k] for k in owned_idx]
            top5_idx = np.argsort(-cos[j])[:5]
            top5 = ", ".join(f"{words[k]} ({cos[j, k]:.3f})"
                             for k in top5_idx)
            lines.append(
                f"[{rank + 1}/16] 融合 token #{j} | 注意力 {attn[j]:.4f} | "
                f"归属词数 {len(owned_idx)}")
            lines.append(f"    最近银行词 top-5: {top5}")
            lines.append(f"    归属词 ({len(owned_idx)}): "
                         + ", ".join(owned_words))
            lines.append("")
    # 人工判定存在 sidecar 文件 (fusion_lens_judgments.txt), 附在机算表后;
    # 脚本重跑不覆盖人工内容。
    sidecar = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fusion_lens_judgments.txt")
    if os.path.exists(sidecar):
        with open(sidecar, encoding="utf-8") as f:
            lines.append("\n" + f.read().rstrip())
    else:
        lines.append("【内容性质判定】待人工复核后登记 (判定依据 = 上方词表)。")

    text = "\n".join(lines)
    out_txt = os.path.join(OUT_DIR, "fusion_lens_profile.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"txt 已写入 {out_txt}")

    make_figure(cos_npz, stats, order_cache={})


def make_figure(cos_npz, stats, order_cache=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, aspect in zip(axes, ASPECTS):
        cos = cos_npz[f"{aspect}_cosine"]
        words = [str(w) for w in cos_npz[f"{aspect}_words"]]
        attn = np.array(stats["caueeg"][aspect]["mean_attn"])
        owner = cos.argmax(axis=0)
        order = np.argsort(-attn)
        y = np.arange(16)[::-1]
        ax.barh(y, attn[order], height=0.6, color="#7c3aed", alpha=0.85)
        for row, j in enumerate(order):
            owned_idx = np.where(owner == j)[0]
            top3_idx = owned_idx[np.argsort(-cos[j, owned_idx])[:3]]
            top3 = " ".join(words[k] for k in top3_idx)
            ax.text(attn[j] + 0.004, y[row],
                    f"{top3}  n={len(owned_idx)}", va="center", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels([f"t{j} · {attn[j]:.3f}" for j in order],
                           fontsize=8)
        ax.set_xlim(0, 0.17)
        ax.set_xlabel("Attention (exp1)")
        ax.set_title(f"{aspect.upper()} fusion tokens: attention and "
                     "owned bank words")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "fusion_lens_profile.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"图已写入 {out_png}")


if __name__ == "__main__":
    main()
