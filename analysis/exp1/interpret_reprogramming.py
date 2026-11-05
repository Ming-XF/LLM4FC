#!/usr/bin/env python
"""KnowledgeLLM 重编程选择性分析 (对应 TimeLLM Fig.5(e) Reprogramming Interpretation).

可视化 EEG patch token x 16 融合 token 的交叉注意力权重 (fc / psd 两模态)：

- 不修改模型代码：在分析脚本内用与 ReprogrammingLayer 完全相同的权重与公式
  复算注意力 (query_projection -> view heads -> key 单位化 -> einsum -> softmax)；
- OFAPatchEncoder 的 replicate 尾部 pad token 去除 (attn[..., :-1, :])：
  fc 实有 1 个 token (画柱状图)，psd 实有 N-1 个 token (画热力图)；
- 同域 (CAUEEG test) 与跨域 (TUAB test) 各 16 正 + 16 负样本，报告：
  每个融合 token 的平均注意力、参与率 (effective token 数)、top-3、跨域相关性；
- 产物：analysis/results/attn_fc_psd.npz + fc_attention.png +
  psd_attention_{caueeg,tuab}.png + stats.json。

运行：sh analysis/exp1/run_interpret.sh
"""

import copy
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CKPT = ("output_dir/output_dir_knowledge_s42_a3a/"
        "KnowledgeLLM_DiseaseCAUEEG_train")
OUT_DIR = "analysis/exp1/results"
N_PER_CLASS = 16

# ── 与训练/评估脚本完全一致的模型参数 (ckpt 形状一致性的唯一来源) ──
sys.argv = [
    "analysis_interpret",
    "--model", "KnowledgeLLM",
    "--dataset", "DiseaseCAUEEG",
    "--batch_size", "16",
    "--d_model", "128",
    "--num_heads", "8",
    "--dropout", "0.1",
    "--llm_type", "llama",
    "--llm_path", "./model/deepseek-r1-distill-llama-8B",
    "--llm_init", "pretrained",
    "--knowledge_bank", "",
    "--knowledge_top_k", "8",
    "--knowledge_top_m", "32",
    "--knowledge_temperature", "1.0",
    "--knowledge_fusion_tokens", "16",
    "--knowledge_raw_patch", "200",
    "--knowledge_raw_stride", "200",
    "--knowledge_psd_patch", "8",
    "--knowledge_psd_stride", "4",
    "--knowledge_fc_patch", "1",
    "--knowledge_fc_stride", "1",
    "--knowledge_llm_layers", "6",
    "--knowledge_modalities", "no_raw",
    "--contrast_lambda", "0.3",
    "--knowledge_contrast_temperature", "0.2",
]

from config import init_config                      # noqa: E402
from main import set_seed                           # noqa: E402
from trainers import KnowledgeLLMTrainer           # noqa: E402
from data import DataConfig, DiseaseTUABDataset    # noqa: E402
from model.KnowledgeLLM.knowledge_bank import ASPECTS  # noqa: E402


def collect_balanced(dataset, n_per_class):
    """从 test 模式数据集收集每类 n_per_class 个样本 (按遇到顺序)。"""
    dataset.mode = 'test'
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    items = {'time_series': [], 'window_fc': [], 'window_psd': [], 'labels': []}
    counts = {0: 0, 1: 0}
    for item in loader:
        lab = item['labels']
        if lab.dim() > 1 and lab.shape[-1] > 1:
            lab = lab.argmax(dim=-1)
        for j in range(lab.shape[0]):
            c = int(lab[j])
            if counts[c] >= n_per_class:
                continue
            counts[c] += 1
            for k in items:
                items[k].append(torch.as_tensor(item[k][j]).clone())
            items['labels'][-1] = c
        if all(v >= n_per_class for v in counts.values()):
            break
    assert all(v >= n_per_class for v in counts.values()), \
        f"收集不足: {counts}"
    batch = {}
    for k in ('time_series', 'window_fc', 'window_psd'):
        batch[k] = torch.stack(items[k]).float()
    batch['labels'] = torch.tensor(items['labels'], dtype=torch.long)
    return batch


@torch.no_grad()
def reprogramming_attention(model, x, window_fc, window_psd):
    """复算 ReprogrammingLayer 的交叉注意力 (B, heads, L, 16)，每模态一张。

    与 model/_knowledge_tokens + ReprogrammingLayer.forward 逐式一致；
    eval 模式下 dropout 为恒等，故此处略去。
    """
    features = model._encode_modalities(x, window_fc, window_psd)
    out = {}
    for aspect in model.config.modalities:
        i = ASPECTS.index(aspect)
        query = model.query_projections[i](features[aspect])
        source = model.knowledge_fusion_tokens[i]          # captured 快照 (16, d_llm)
        reprog = model.reprogrammers[i]
        b, length, _ = query.shape
        n_proto, _ = source.shape
        q = reprog.query_projection(query).view(
            b, length, reprog.n_heads, reprog.head_dim)
        k = F.normalize(reprog.key_projection(source).view(
            n_proto, reprog.n_heads, reprog.head_dim), dim=-1)
        scores = torch.einsum("blhe,she->bhls", q, k) * \
            (1.0 / math.sqrt(reprog.head_dim))
        attn = torch.softmax(scores, dim=-1)               # (B, heads, L, 16)
        out[aspect] = attn.detach().cpu().numpy()
    return out


def token_stats(attn):
    """attn (S, heads, L, 16) -> 融合 token 平均权重 + 选择性统计。"""
    p = attn.mean(axis=(0, 1, 2))          # (16,)
    p = p / p.sum()
    effective = 1.0 / np.sum(p ** 2)       # 有效 token 数 (<=16)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    top3 = np.argsort(-p)[:3].tolist()
    return p, effective, entropy, top3


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    args = init_config()
    # 与 main.py 一致的 RNG 初始化：启动时的知识银行抽取依赖全局种子
    set_seed(42)
    trainer = KnowledgeLLMTrainer(args, local_rank=0, task_id=0)
    trainer.load_model(path=CKPT)
    model = trainer.model
    model.eval()
    assert model.use_captured_fusion, "captured fusion snapshot 未生效"
    device = next(model.parameters()).device

    # ── 同域: CAUEEG test ──
    caueeg = collect_balanced(trainer.dataset, N_PER_CLASS)
    # ── 跨域: TUAB test (独立 DataConfig，与评估脚本口径一致) ──
    args2 = copy.deepcopy(args)
    args2.dataset = 'DiseaseTUAB'
    tuab_ds = DiseaseTUABDataset(DataConfig(args2), k=0, episode_seed=None)
    tuab = collect_balanced(tuab_ds, N_PER_CLASS)

    results = {}
    stats = {}
    for name, batch in (('caueeg', caueeg), ('tuab', tuab)):
        attn = reprogramming_attention(
            model,
            batch['time_series'].to(device),
            batch['window_fc'].to(device),
            batch['window_psd'].to(device))
        # ── 去 OFAPatchEncoder 的 replicate 尾部 pad token ──
        for aspect in list(attn):
            padded = attn[aspect].shape[2]
            attn[aspect] = attn[aspect][..., :-1, :]
            print(f"[{name}] {aspect}: padded L={padded} -> real "
                  f"L={attn[aspect].shape[2]} (heads={attn[aspect].shape[1]})")
            assert attn[aspect].shape[2] >= 1
        results[name] = attn
        stats[name] = {}
        for aspect in attn:
            p, eff, ent, top3 = token_stats(attn[aspect])
            stats[name][aspect] = {
                'effective_tokens': round(float(eff), 3),
                'entropy': round(ent, 3),
                'top3_tokens': top3,
                'mean_attn': [round(float(v), 4) for v in p],
            }
        print(f"[{name}] {stats[name]}")

    # ── 跨域融合 token 使用一致性 (平均注意力向量的相关) ──
    for aspect in ('fc', 'psd'):
        r = np.corrcoef(
            stats['caueeg'][aspect]['mean_attn'],
            stats['tuab'][aspect]['mean_attn'])[0, 1]
        stats['cross_domain_corr'] = stats.get('cross_domain_corr', {})
        stats['cross_domain_corr'][aspect] = round(float(r), 3)
        print(f"[cross] {aspect} 平均注意力跨域相关 r={r:.3f}")

    np.savez(
        os.path.join(OUT_DIR, 'attn_fc_psd.npz'),
        caueeg_fc=results['caueeg']['fc'],
        caueeg_psd=results['caueeg']['psd'],
        tuab_fc=results['tuab']['fc'],
        tuab_psd=results['tuab']['psd'],
        caueeg_labels=caueeg['labels'].numpy(),
        tuab_labels=tuab['labels'].numpy())
    with open(os.path.join(OUT_DIR, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"npz + stats.json 已写入 {OUT_DIR}")

    make_figures(results, stats)


def make_figures(results, stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 10,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    COLOR = {'caueeg': '#0E7490', 'tuab': '#B45309'}
    DOMAIN_LABEL = {'caueeg': 'CAUEEG (同域)', 'tuab': 'TUAB (跨域)'}

    # ── FC: 单个 patch token -> 16 融合 token 柱状图 (两域并列) ──
    fc = {d: results[d]['fc'].mean(axis=(0, 1))[0] for d in results}   # (16,)
    fc_std = {d: results[d]['fc'].mean(axis=2).reshape(-1, 16).std(axis=0)
              for d in results}
    x = np.arange(16)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for k, d in enumerate(results):
        ax.bar(x + (k - 0.5) * 0.38, fc[d], width=0.38,
               yerr=fc_std[d], capsize=2, color=COLOR[d],
               label=DOMAIN_LABEL[d])
    ax.set_xticks(x)
    ax.set_xlabel('FC 融合 token 索引 (16)')
    ax.set_ylabel('平均交叉注意力')
    ax.set_title('FC 重编程注意力 (单 patch token)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fc_attention.png'), dpi=200,
                bbox_inches='tight')
    plt.close(fig)

    # ── PSD: patch token x 16 融合 token 热力图 (每域一张) ──
    for d in results:
        mat = results[d]['psd'].mean(axis=(0, 1))          # (L, 16)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        im = ax.imshow(mat, aspect='auto', cmap='viridis')
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=7,
                        color='white' if v > np.percentile(mat, 60) else '0.2')
        ax.set_xticks(np.arange(16))
        ax.set_xlabel('PSD 融合 token 索引 (16)')
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_ylabel('PSD patch token')
        ax.set_title(f'PSD 重编程注意力 — {DOMAIN_LABEL[d]}')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'psd_attention_{d}.png'), dpi=200,
                    bbox_inches='tight')
        plt.close(fig)

    print(f"图片已写入 {OUT_DIR}")


if __name__ == '__main__':
    main()
