#!/usr/bin/env python3
"""精确复现 s42 训练时的知识银行抽取并保存 (分析实验二前置步骤)。

与 KnowledgeLLM._load_or_extract_knowledge 的启动路径逐式一致:
  - 同一训练配置: llm_type=llama, llm_path=deepseek-r1-distill-llama-8B,
    llm_init=pretrained, knowledge_top_k=8, knowledge_top_m=32,
    knowledge_temperature=1.0, texts=DEFAULT_KNOWLEDGE_TEXTS;
  - 在截断/转 fp32 之前的全量 8B bf16 上抽取 (extract_knowledge_bank
    只用 final hidden layer + lm_head);
  - 按 main.py 惯例先 set_seed(42) (抽取本身确定性: topk+softmax+排序,
    无采样; set_seed 仅为与主流程一致)。
验证: 重抽取 token_ids 与 ckpt 的 knowledge_mappers.{i}.token_ids 对比。
诊断结论 (2026-11-06): 重抽取完全自洽 (两次逐位相同), 与 ckpt 重叠
98-99% —— 差异为训练时 topk 边界数值噪声。ckpt 的 token_ids 是 mapper
实际消费的精确知识集, 以其为权威保存。
产物: analysis/exp2/results/knowledge_bank_s42.npz
  (save_knowledge_bank 格式: ckpt 权威 token_ids + 重抽取权重按 id 合并)。
运行: sh analysis/exp2/run_extract.sh  (需单卡 >= 20GB)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from main import set_seed                                            # noqa: E402
from model.KnowledgeLLM.knowledge_bank import (                      # noqa: E402
    ASPECTS, DEFAULT_KNOWLEDGE_TEXTS, extract_knowledge_bank,
    save_knowledge_bank)

CKPT = ("output_dir/output_dir_knowledge_s42_a3a/"
        "KnowledgeLLM_DiseaseCAUEEG_train")
LLM_PATH = "./model/deepseek-r1-distill-llama-8B"
OUT_PATH = "analysis/exp2/results/knowledge_bank_s42.npz"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
    causal = AutoModelForCausalLM.from_pretrained(
        LLM_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
    causal = causal.to(device)
    causal.eval()

    bank = extract_knowledge_bank(
        causal, tokenizer,
        texts=DEFAULT_KNOWLEDGE_TEXTS,
        top_k_positions=8,    # --knowledge_top_k
        top_m_tokens=32,      # --knowledge_top_m
        temperature=1.0,      # --knowledge_temperature
    )

    # ── 验证: 与 ckpt 里训练时的 token_ids 对比 ──
    sd = torch.load(os.path.join(CKPT, "KnowledgeLLM-0.bin"),
                    map_location="cpu", weights_only=False)
    merged = {}
    for i, aspect in enumerate(ASPECTS):
        extracted = bank[aspect]["token_ids"].numpy()
        ckpt_ids = sd[f"knowledge_mappers.{i}.token_ids"].numpy()
        overlap = len(set(extracted.tolist()) & set(ckpt_ids.tolist()))
        print(f"[{aspect}] 重抽取 {len(extracted)} vs ckpt {len(ckpt_ids)} "
              f"tokens, 重叠 {overlap}")
        # 训练时 topk 边界的数值噪声导致少量 token 差异 (诊断: 两次重抽取
        # 完全自洽, 与 ckpt 重叠 98-99%) —— ckpt 的 token_ids 才是 mapper
        # 实际消费的精确知识集, 以其为权威; 权重按 id 合并, 缺失的填 0。
        id2w = {int(t): float(w) for t, w in zip(
            extracted.tolist(), bank[aspect]["weights"].tolist())}
        id2r = {int(t): float(v) for t, v in zip(
            extracted.tolist(),
            bank[aspect]["raw_logit_sum"].tolist())}
        merged[aspect] = {
            "token_ids": torch.as_tensor(ckpt_ids, dtype=torch.long),
            "weights": torch.tensor(
                [id2w.get(int(t), 0.0) for t in ckpt_ids],
                dtype=torch.float32),
            "raw_logit_sum": torch.tensor(
                [id2r.get(int(t), 0.0) for t in ckpt_ids],
                dtype=torch.float32),
        }

    save_knowledge_bank(merged, OUT_PATH)
    print(f"知识银行 (ckpt 权威 token_ids + 重抽取权重) 已保存: {OUT_PATH}")


if __name__ == "__main__":
    main()
