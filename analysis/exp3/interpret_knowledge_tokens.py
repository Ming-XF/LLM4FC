#!/usr/bin/env python3
"""知识 token 语义分析 (分析实验三, 试验一)。

回答: 进 LLM 的"知识 token"(每样本 22 个 = FC 1 + PSD 21 的
reprogramming 输出向量)在语言空间里对应什么 — 用 LLM 词表嵌入的余弦
相似度取 top-5 词表示其语义 ("LLM 实际读到的文本")。

- 不修改模型代码: 用训练/评估完全相同的路径取知识 token
  (model._encode_modalities + model._knowledge_tokens,
   use_captured_fusion 快照, eval 模式);
- case: CAUEEG/TUAB test 各取 2 正 + 2 负 = 8 个样本 (按遇到顺序);
- PSD 的 OFA replicate 尾部 pad token 去除 (同 exp1 口径, [:, :-1]);
- 词表 = LLM 输入嵌入矩阵 embed_tokens (与 mapper/LLM 输入同一空间);
  余弦 top-5 排除特殊 token;
- 产物: results/knowledge_token_semantics.txt +
  results/knowledge_token_semantics.png (8 样本 x 22 位置, 每格 top-1 词,
  底色 = 相似度) + results/knowledge_tokens.npz (向量/top5/相似度)。

运行: sh analysis/exp3/run_interpret.sh
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

CKPT = ("output_dir/output_dir_knowledge_s42_a3a/"
        "KnowledgeLLM_DiseaseCAUEEG_train")
LLM_PATH = "./model/deepseek-r1-distill-llama-8B"
LLM_SHARD = ("model/deepseek-r1-distill-llama-8B/"
             "model-00001-of-000002.safetensors")
OUT_DIR = "analysis/exp3/results"
N_PER_CLASS = 2

# ── 与训练/评估脚本完全一致的模型参数 (ckpt 形状一致性的唯一来源) ──
sys.argv = [
    "analysis_exp3",
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


def collect_balanced(dataset, n_per_class):
    """从 test 模式数据集收集每类 n_per_class 个样本 (按遇到顺序)。"""
    dataset.mode = 'test'
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    items = {'time_series': [], 'window_fc': [], 'window_psd': [],
             'labels': []}
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
def knowledge_tokens(model, x, window_fc, window_psd):
    """与评估路径一致的知识 token: (B, 22, 4096) = FC 1 + PSD 21。"""
    features = model._encode_modalities(x, window_fc, window_psd)
    knowledge = model._knowledge_tokens(features)      # [fc, psd] 各 (B,L,4096)
    for i in range(len(knowledge)):
        knowledge[i] = knowledge[i][:, :-1, :]         # 去 OFA pad token
    return torch.cat(knowledge, dim=1)                 # (B, 22, 4096)


def decode_surface(tokenizer, token_id):
    tok = tokenizer.convert_ids_to_tokens(int(token_id))
    tok = tok.replace("Ġ", " ").replace("▁", " ")
    return tok.strip() or f"<{token_id}>"


def load_embedding_weight():
    from safetensors import safe_open
    with safe_open(LLM_SHARD, framework="pt", device="cpu") as f:
        return f.get_tensor("model.embed_tokens.weight").float()   # (V, 4096)


def main():
    from transformers import AutoTokenizer

    os.makedirs(OUT_DIR, exist_ok=True)
    args = init_config()
    set_seed(42)
    trainer = KnowledgeLLMTrainer(args, local_rank=0, task_id=0)
    trainer.load_model(path=CKPT)
    model = trainer.model
    model.eval()
    assert model.use_captured_fusion, "captured fusion snapshot 未生效"
    device = next(model.parameters()).device

    caueeg = collect_balanced(trainer.dataset, N_PER_CLASS)
    args2 = args
    import copy
    args2 = copy.deepcopy(args)
    args2.dataset = 'DiseaseTUAB'
    tuab_ds = DiseaseTUABDataset(DataConfig(args2), k=0, episode_seed=None)
    tuab = collect_balanced(tuab_ds, N_PER_CLASS)

    # ── 知识 token 向量 ──
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
    E = load_embedding_weight()
    special = set(int(i) for i in tokenizer.all_special_ids if i is not None)
    allowed = torch.ones(E.shape[0], dtype=torch.bool)
    for i in special:
        if i < E.shape[0]:
            allowed[i] = False

    all_vecs, all_labels = [], []
    for name, batch in (('caueeg', caueeg), ('tuab', tuab)):
        vecs = knowledge_tokens(
            model,
            batch['time_series'].to(device),
            batch['window_fc'].to(device),
            batch['window_psd'].to(device)).detach().float().cpu()
        all_vecs.append(vecs)
        all_labels.append(batch['labels'])
    vecs = torch.cat(all_vecs, dim=0)                  # (8, 22, 4096)
    labels = torch.cat(all_labels, dim=0)              # (8,)
    domains = ['caueeg'] * 4 + ['tuab'] * 4
    n_samples, n_tok, d = vecs.shape

    # ── 词表余弦 top-5 ──
    vn = F.normalize(vecs.reshape(-1, d), dim=-1)      # (176, 4096)
    En = F.normalize(E, dim=-1)                        # (152064, 4096)
    sim = vn @ En.T                                    # (176, 152064)
    sim[:, ~allowed] = -float('inf')
    topv, topi = sim.topk(5, dim=-1)                   # (176, 5)

    lines = []
    lines.append("实验三: 知识 token 语义分析 (s42 a3a, CAUEEG 训练)")
    lines.append("口径: 每样本 22 个知识 token (FC 1 + PSD 21), "
                 "LLM 输入词表余弦 top-5 (排除特殊 token)\n")
    pos_labels = {0: 'FC', 1: 'PSD'}
    for s in range(n_samples):
        lines.append(f"[{domains[s]} test] 样本 {s % 4 + 1} (标签 {labels[s]})")
        for t in range(n_tok):
            idx = s * n_tok + t
            mod = pos_labels[0 if t == 0 else 1]
            pos = t if t == 0 else t
            words = ", ".join(
                f"{decode_surface(tokenizer, topi[idx][k])} "
                f"({topv[idx][k]:.3f})" for k in range(5))
            lines.append(f"  {mod} #{pos:>2}  {words}")
        lines.append("")

    text = "\n".join(lines)
    out_txt = os.path.join(OUT_DIR, 'knowledge_token_semantics.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text[:1500])
    print(f"...\ntxt 已写入 {out_txt}")

    np.savez(os.path.join(OUT_DIR, 'knowledge_tokens.npz'),
             vecs=vecs.numpy(), labels=labels.numpy(),
             domains=np.array(domains),
             top5_ids=topi.numpy(), top5_sims=topv.numpy(),
             top1_sims=sim.max(dim=-1).values.numpy().reshape(n_samples, n_tok))
    make_figure(vecs, labels, domains, tokenizer, topi, topv, sim)
    print(f"产物已写入 {OUT_DIR}")


def make_figure(vecs, labels, domains, tokenizer, topi, topv, sim):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_samples, n_tok, _ = vecs.shape
    grid = sim.max(dim=-1).values.numpy().reshape(n_samples, n_tok)
    words = [[decode_surface(tokenizer, topi[s * n_tok + t][0])
              for t in range(n_tok)] for s in range(n_samples)]

    fig, ax = plt.subplots(figsize=(15, 5.2))
    im = ax.imshow(grid, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
    thr = np.percentile(grid, 60)
    for s in range(n_samples):
        for t in range(n_tok):
            w = words[s][t]
            if len(w) > 8:
                w = w[:7] + '…'
            ax.text(t, s, w, ha='center', va='center', fontsize=6.5,
                    color='white' if grid[s, t] > thr else '0.2')
    ax.set_yticks(np.arange(n_samples))
    ax.set_yticklabels([f"{domains[s]} {'P' if labels[s] else 'N'}#"
                        f"{s % 4 + 1}" for s in range(n_samples)], fontsize=9)
    ax.set_xticks(np.arange(n_tok))
    ax.set_xticklabels(['FC'] + [f'PSD{i}' for i in range(1, n_tok)],
                       rotation=90, fontsize=6)
    ax.set_title("Knowledge tokens entering the LLM: top-1 vocabulary word "
                 "per position (s42 a3a)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02,
                 label='cosine similarity')
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, 'knowledge_token_semantics.png')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"图已写入 {out_png}")


if __name__ == '__main__':
    main()
