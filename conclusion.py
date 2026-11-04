#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conclusion.py
=============
汇总六模型实验结果, 渲染 HTML 对比表格 (默认 conclusion.html)。

口径:
  - 主表拆为两张: 同域表 (训练域 → 同域测试) 与跨域零样本表
    (CAUEEG/TUAB 互转); TUEP 已移出 (全方法接近随机, 作阴性对照)。
  - 每格报告三项指标: Balanced Acc / AUC-ROC / AUC-PR
    (百分数刻度, 3 seeds mean ± std); 各数值可溯源至对应文件
    (metrics JSON / 结果 JSON / 评估日志, 见悬浮提示)。
  - KnowledgeLLM: 主方法 (10s FC+PSD 双模态 [no_raw] + 知识银行 +
    冻结 6 层 Llama), CAUEEG/TUAB 双训练域, 3 个 seed (42/43/44);
    指标取自 log_dir/metrics_KnowledgeLLM_*.json。
  - EEGNet / OFA: 指标取自 log_dir/metrics_{EEGNet,OFA}_*.json
    (重测落盘, AUC 与评估日志逐位一致); EEG-Deformer 取自
    baseline/EEG-Deformer/save_llm4fc[_s43|_s44]/results/*.json;
    NeuroLM 取自 test_results.json; LaBraM 取自评估日志 Global 行
    与同域 log.txt。均为 10s 段输入 (每 60s 窗取首个 10s 段),
    3 个 seed (42/43/44)。
  - 消融表 = 主方法 no_raw (CAUEEG 训练域) vs 各消融变体, 3 个 seed
    (42/43/44), 同域 CAUEEG 与 →TUAB 零样本两格, window AUC-ROC
    mean ± std (变体未重测新指标); Δ = 与主方法 mean 之差。注意:
    A1/A2a/A3c/A3b/A4/A5 消融变体基于含 raw 的三模态设置运行
    (设置差异后续说明)。
    变体日志命名约定:
    训练 train_KnowledgeLLM_s{seed}_caueeg_{variant}_DiseaseCAUEEG.log,
    零样本 test_KnowledgeLLM_s{seed}_{variant}_caueeg_to_tuab_zero-shot_DiseaseTUAB.log;
    新增变体在 ABLATION_VARIANTS 登记 (key, label) 即并入表格。

用法:
  python conclusion.py [--out conclusion.html]
"""

import argparse
import json
import re
import sys
from datetime import datetime
from html import escape
from pathlib import Path

REPO = Path(__file__).resolve().parent
LOG_DIR = REPO / "log_dir"
# 主表双域 (TUEP 全方法随机, 已移出主表作阴性对照)
DOMAINS = ["CAUEEG", "TUAB"]
SEEDS = ["42", "43", "44"]

# ─────────────────────────────────────────────────────────────────────────
# 解析: 主方法 KnowledgeLLM / EEGNet / OFA —— 统一从 log_dir 的
# metrics_{model}_s{seed}_{tag}_Disease{D}.json 读取 (BalancedAccuracy /
# AUC / AUC_PR), 每个数值可溯源到对应 JSON 文件 (其 AUC/AUC_PR 与
# 评估日志逐位一致)。
# ─────────────────────────────────────────────────────────────────────────

METRICS_JSON_TAG_MAP = {
    "caueeg_test": ("CAUEEG", "CAUEEG"),
    "tuab_test": ("TUAB", "TUAB"),
    "caueeg_to_tuab": ("CAUEEG", "TUAB"),
    "tuab_to_caueeg": ("TUAB", "CAUEEG"),
}


def _json_metrics(rec):
    m = rec.get("metrics", rec)
    return {"auc": float(m["AUC"]),
            "auc_pr": float(m["AUC_PR"]),
            "bal_acc": float(m["BalancedAccuracy"]),
            "accuracy": float(m.get("Accuracy", 0.0))}


def discover_from_metrics_jsons(model, prefix):
    exps = []
    for p in sorted(LOG_DIR.glob(f"metrics_{prefix}_s*_*.json")):
        m = re.match(
            rf"metrics_{prefix}_s(\d+)_(?:a3a_)?(caueeg_test|tuab_test|"
            r"caueeg_to_tuab|tuab_to_caueeg)_Disease\w+\.json", p.name)
        if not m:
            continue
        seed, tag = m.groups()
        if seed not in SEEDS:
            continue
        src, tgt = METRICS_JSON_TAG_MAP[tag]
        exps.append({
            "model": model, "seed": seed, "source": src, "target": tgt,
            "mode": "同域" if src == tgt else "零样本",
            "metrics": _json_metrics(json.loads(p.read_text(encoding="utf-8"))),
            "src": p,
        })
    return exps


def discover_knowledgellm():
    return discover_from_metrics_jsons("KnowledgeLLM", "KnowledgeLLM")


# ─────────────────────────────────────────────────────────────────────────
# 解析: KnowledgeLLM 消融变体 (CAUEEG 训练域)
#   同域: train_KnowledgeLLM_s{seed}_caueeg_{variant}_DiseaseCAUEEG.log
#   零样本: test_KnowledgeLLM_s{seed}_{variant}_caueeg_to_tuab_zero-shot_DiseaseTUAB.log
#   定稿方法本身无 variant 标记, 其两格直接复用主表 cells
#   [(KnowledgeLLM, CAUEEG, {CAUEEG,TUAB})], 保证与主表数字一致。
# ─────────────────────────────────────────────────────────────────────────

ABLATION_VARIANTS = [
    ("a1", "A1 随机知识银行"),
    ("a2a", "A2a LLM随机+预训练bank"),
    ("a3c", "A3c 去掉fc (raw+PSD)"),
    ("a3b", "A3b 去掉psd (raw+FC)"),
    ("a4", "A4 无跨注意力 (静态知识相加)"),
    ("a5", "A5 无对比损失"),
]

ABL_TRAIN_RE = re.compile(
    r"train_KnowledgeLLM_s(\d+)_caueeg_([a-z0-9]+)_DiseaseCAUEEG\.log")
ABL_ZS_RE = re.compile(
    r"test_KnowledgeLLM_s(\d+)_([a-z0-9]+)_caueeg_to_tuab_zero-shot_DiseaseTUAB\.log")


def discover_ablations():
    """variant -> {seed -> {'same': (metrics, src), 'zs': (metrics, src)}}"""
    exps = {}
    for p in sorted(LOG_DIR.glob(
            "train_KnowledgeLLM_s*_caueeg_*_DiseaseCAUEEG.log")):
        m = ABL_TRAIN_RE.match(p.name)
        if not m:
            continue
        seed, variant = m.groups()
        if seed not in SEEDS:
            continue
        if variant == "a3a":
            continue  # a3a = no_raw 主方法, 已入主表, 不进消融表
        met = parse_trainer_log(p)
        if met:
            exps.setdefault(variant, {}).setdefault(
                seed, {})["same"] = (met, p)
    for p in sorted(LOG_DIR.glob(
            "test_KnowledgeLLM_s*_*_caueeg_to_tuab_zero-shot_DiseaseTUAB.log")):
        m = ABL_ZS_RE.match(p.name)
        if not m:
            continue
        seed, variant = m.groups()
        if seed not in SEEDS:
            continue
        if variant == "a3a":
            continue
        met = parse_trainer_log(p)
        if met:
            exps.setdefault(variant, {}).setdefault(seed, {})["zs"] = (met, p)
    return exps


# ─────────────────────────────────────────────────────────────────────────
# 解析: EEGNet (主框架 trainer 日志)
#   s42: train_EEGNet_Disease{D}.log / test_EEGNet_{src}_zero-shot_Disease{T}.log
#   s43/s44: train_EEGNet_s{seed}_Disease{D}.log / test_EEGNet_s{seed}_{src}_...
# ─────────────────────────────────────────────────────────────────────────

METRIC_LINE_RE = re.compile(
    r"^\S+ \S+ - INFO - \S+ -\s+(Loss|AUC|Accuracy|Precision|Specificity|Sensitivity): ([0-9.eE+-]+)"
)
SRC_TAGS = {"caueeg": "CAUEEG", "tuab": "TUAB", "tuep": "TUEP"}


def parse_trainer_log(path):
    blocks, cur = [], {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = METRIC_LINE_RE.match(line)
        if m:
            cur[m.group(1).lower()] = float(m.group(2))
        elif cur:
            blocks.append(cur)
            cur = {}
    if cur:
        blocks.append(cur)
    return blocks[-1] if blocks else None


def discover_eegnet():
    return discover_from_metrics_jsons("EEGNet", "EEGNet")


# ─────────────────────────────────────────────────────────────────────────
# 解析: EEG-Deformer (save_llm4fc[_s43|_s44]/results/*.json, sklearn AUC)
# ─────────────────────────────────────────────────────────────────────────

DEFORMER_DIRS = {"42": REPO / "baseline" / "EEG-Deformer" / "save_llm4fc",
                 "43": REPO / "baseline" / "EEG-Deformer" / "save_llm4fc_s43",
                 "44": REPO / "baseline" / "EEG-Deformer" / "save_llm4fc_s44"}
DEFORMER_NAME_RE = re.compile(r"Disease(\w+)__(\w+)\.json")


def discover_deformer():
    exps = []
    for seed, base in DEFORMER_DIRS.items():
        d = base / "results"
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            m = DEFORMER_NAME_RE.match(p.name)
            if not m:
                continue
            src, tgt = m.groups()
            if src not in DOMAINS or tgt not in DOMAINS:
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            exps.append({
                "model": "EEG-Deformer", "seed": seed, "source": src,
                "target": tgt, "mode": "同域" if src == tgt else "零样本",
                "metrics": {"auc": float(data["auc"]),
                            "auc_pr": float(data["pr_auc"]),
                            "bal_acc": float(data["balanced_accuracy"]),
                            "accuracy": float(data["accuracy"])},
                "src": p,
            })
    return exps


# ─────────────────────────────────────────────────────────────────────────
# 解析: LaBraM
#   零样本: log_llm4fc/out_llm4fc[_s43|_s44]_{src}__Disease{T}.log 的 Global 行
#   同域: out_llm4fc[_s43|_s44]_{dom}/log.txt 最后一行 JSON (test_roc_auc)
# ─────────────────────────────────────────────────────────────────────────

LABRAM_BASE = REPO / "baseline" / "LaBraM"
LABRAM_SRC = {"caueeg": "CAUEEG", "tuab": "TUAB", "tuep": "TUEP"}
LABRAM_EVAL_RE = re.compile(r"out_llm4fc(_s\d+)?_(\w+)__Disease(\w+)\.log")
LABRAM_TRAIN_DIRS = {
    "42": {"CAUEEG": "out_llm4fc_caueeg", "TUAB": "out_llm4fc_tuab",
           "TUEP": "out_llm4fc_tuep"},
    "43": {"CAUEEG": "out_llm4fc_s43_caueeg", "TUAB": "out_llm4fc_s43_tuab",
           "TUEP": "out_llm4fc_s43_tuep"},
    "44": {"CAUEEG": "out_llm4fc_s44_caueeg", "TUAB": "out_llm4fc_s44_tuab",
           "TUEP": "out_llm4fc_s44_tuep"},
}


def discover_labram():
    exps = []
    covered = set()  # (seed, src, tgt) 已由 eval 日志覆盖
    evaldir = LABRAM_BASE / "log_llm4fc"
    if evaldir.is_dir():
        for p in sorted(evaldir.glob("*.log")):
            m = LABRAM_EVAL_RE.match(p.name)
            if not m:
                continue
            seed = m.group(1)[2:] if m.group(1) else "42"
            src = LABRAM_SRC.get(m.group(2))
            tgt = m.group(3)
            if src is None or tgt not in DOMAINS:
                continue
            auc = pr_auc = bal_acc = acc = None
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                g = re.search(r"======Global: (\{.*\})", line)
                if g:
                    d = json.loads(g.group(1))
                    if d.get("roc_auc") is not None:
                        auc = float(d["roc_auc"])
                    if d.get("pr_auc") is not None:
                        pr_auc = float(d["pr_auc"])
                    if d.get("balanced_accuracy") is not None:
                        bal_acc = float(d["balanced_accuracy"])
                    if d.get("accuracy") is not None:
                        acc = float(d["accuracy"])
            if auc is None:
                continue
            # 同域与跨域统一走 best-ckpt 的 --eval 日志 (Global 行)
            exps.append({
                "model": "LaBraM", "seed": seed, "source": src, "target": tgt,
                "mode": "同域" if src == tgt else "零样本",
                "metrics": {"auc": auc, "auc_pr": pr_auc,
                            "bal_acc": bal_acc, "accuracy": acc},
                "src": p,
            })
            covered.add((seed, src, tgt))
    # 兜底: 无 eval 日志的同域格回退训练 log.txt 最后一行 (末轮 test)
    for seed, dirs in LABRAM_TRAIN_DIRS.items():
        for dom, dname in dirs.items():
            if dom not in DOMAINS:
                continue
            if (seed, dom, dom) in covered:
                continue
            logf = LABRAM_BASE / dname / "log.txt"
            if not logf.exists():
                continue
            lines = logf.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in reversed(lines):
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("test_roc_auc") is not None:
                    exps.append({
                        "model": "LaBraM", "seed": seed, "source": dom,
                        "target": dom, "mode": "同域",
                        "metrics": {"auc": float(d["test_roc_auc"]),
                                    "auc_pr": float(d["test_pr_auc"]),
                                    "bal_acc": float(
                                        d["test_balanced_accuracy"]),
                                    "accuracy": float(d["test_accuracy"])},
                        "src": logf,
                    })
                    break
    return exps


# ─────────────────────────────────────────────────────────────────────────
# 解析: NeuroLM (out_llm4fc[_s43|_s44][_{tuab,tuep}]/test_results.json)
# ─────────────────────────────────────────────────────────────────────────

NEUROLM_BASE = REPO / "baseline" / "NeuroLM"
NEUROLM_DIRS = {
    "42": [("CAUEEG", "out_llm4fc"), ("TUAB", "out_llm4fc_tuab"),
           ("TUEP", "out_llm4fc_tuep")],
    "43": [("CAUEEG", "out_llm4fc_s43"), ("TUAB", "out_llm4fc_s43_tuab"),
           ("TUEP", "out_llm4fc_s43_tuep")],
    "44": [("CAUEEG", "out_llm4fc_s44"), ("TUAB", "out_llm4fc_s44_tuab"),
           ("TUEP", "out_llm4fc_s44_tuep")],
}


def discover_neurolm():
    exps = []
    for seed, dirs in NEUROLM_DIRS.items():
        for src, dname in dirs:
            p = NEUROLM_BASE / dname / "test_results.json"
            if not p.exists():
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            for dom in DOMAINS:
                if dom not in data or not isinstance(data[dom], dict):
                    continue
                w = data[dom]["test"]["window"]
                exps.append({
                    "model": "NeuroLM", "seed": seed, "source": src,
                    "target": dom, "mode": "同域" if dom == src else "零样本",
                    "metrics": {"auc": float(w["roc_auc"]),
                                "auc_pr": float(w["pr_auc"]),
                                "bal_acc": float(w["balanced_accuracy"]),
                                "accuracy": float(w["accuracy"])},
                    "src": p,
                })
    return exps


# ─────────────────────────────────────────────────────────────────────────
# 解析: OFA (experiments_llm4fc/*/output.log 的 Test Summary; seed 来自
#       configuration.json; 目录名 Disease{D}_10s_* = 同域评估, ZS_{S}_{T}_*
#       = 零样本)
# ─────────────────────────────────────────────────────────────────────────

def discover_ofa():
    return discover_from_metrics_jsons("OFA", "OFA")


# ─────────────────────────────────────────────────────────────────────────
# 聚合 + 渲染
# ─────────────────────────────────────────────────────────────────────────

MODELS = ["EEGNet", "NeuroLM", "LaBraM", "OFA", "EEG-Deformer",
          "KnowledgeLLM"]
MODEL_COLORS = {
    "EEGNet": "#2563eb", "KnowledgeLLM": "#7c3aed", "NeuroLM": "#059669",
    "LaBraM": "#b45309", "OFA": "#db2777", "EEG-Deformer": "#0891b2",
}


def _fmt(x, nd=2):
    """指标统一以百分数刻度显示 (原始 0-1 值 × 100, 不带 % 号)。"""
    return f"{x * 100:.{nd}f}" if x is not None else "—"


def _cell_text(vals):
    if len(vals) == 1:
        return _fmt(vals[0])
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    std = var ** 0.5
    return f"{_fmt(mean)} ± {_fmt(std)}"


def _cell_tooltip(model, source, target, cell, key):
    lines = [f"{model} | {source} → {target} ({cell['mode']})"]
    for seed, met, src in cell["seeds"]:
        v = met.get(key)
        if v is not None:
            lines.append(f"seed {seed}: {key}={_fmt(v)}")
        lines.append(f"  来源: {src}")
    if len(cell["seeds"]) > 1:
        vals = [m[key] for _, m, _ in cell["seeds"]
                if m.get(key) is not None]
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            lines.append(f"mean ± std: {_fmt(mean)} ± {_fmt(var ** 0.5)}")
    return escape("\n".join(lines))


def build_html():
    # 收集
    exps = (discover_knowledgellm() + discover_eegnet()
            + discover_deformer() + discover_labram() + discover_neurolm()
            + discover_ofa())

    # 聚合: (model, source, target) -> {mode, seeds: [(seed, metrics, src)]}
    cells = {}
    for e in exps:
        key = (e["model"], e["source"], e["target"])
        c = cells.setdefault(key, {"mode": e["mode"], "seeds": []})
        c["seeds"].append((e["seed"], e["metrics"], e["src"]))

    same_pairs = [(s, s) for s in DOMAINS]
    cross_pairs = [(s, t) for s in DOMAINS for t in DOMAINS if t != s]
    # 每格三项指标: Balanced Acc / AUC-ROC / AUC-PR
    METRIC_COLS = [("bal_acc", "BalAcc"), ("auc", "AUC-ROC"),
                   ("auc_pr", "AUC-PR")]

    def _metrics_table(title, pairs, cells):
        rows = []
        for model in MODELS:
            row = {}
            for p in pairs:
                row[p] = cells.get((model, p[0], p[1]))
            rows.append((model, row))

        best = {p: {k: None for k, _ in METRIC_COLS} for p in pairs}
        for _, row in rows:
            for p, c in row.items():
                if not c:
                    continue
                for key, _ in METRIC_COLS:
                    vals = [m.get(key) for _, m, _ in c["seeds"]]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        mean = sum(vals) / len(vals)
                        cur = best[p][key]
                        if cur is None or mean > cur:
                            best[p][key] = mean

        thead1 = ['<th rowspan="2" class="left">模型</th>']
        thead2 = []
        for src, tgt in pairs:
            thead1.append(f'<th colspan="3">{escape(src)} → {escape(tgt)}</th>')
            thead2.append("<th>BalAcc</th><th>AUC-ROC</th><th>AUC-PR</th>")

        tbody = []
        for model, row in rows:
            color = MODEL_COLORS.get(model, "#666")
            tr = [f'<td class="model" style="--mc:{color}">{escape(model)}</td>']
            for p in pairs:
                c = row[p]
                if c:
                    for key, _ in METRIC_COLS:
                        vals = [m.get(key) for _, m, _ in c["seeds"]]
                        vals = [v for v in vals if v is not None]
                        if not vals:
                            tr.append('<td class="empty">—</td>')
                            continue
                        mean = sum(vals) / len(vals)
                        style = (' class="best"'
                                 if best[p][key] is not None
                                 and abs(mean - best[p][key]) < 1e-9 else "")
                        tr.append(
                            f'<td class="cell" title="{_cell_tooltip(model, p[0], p[1], c, key)}">'
                            f'<span{style}>{_cell_text(vals)}</span></td>')
                else:
                    tr.append('<td class="empty" colspan="3">—</td>')
            tbody.append("<tr>" + "".join(tr) + "</tr>")

        return (f'<h2>{escape(title)}</h2>\n'
                f'<table>\n  <thead>\n    <tr>{"".join(thead1)}</tr>\n'
                f'    <tr>{"".join(thead2)}</tr>\n  </thead>\n'
                f'  <tbody>\n{chr(10).join(tbody)}\n  </tbody>\n</table>')

    tables = (_metrics_table("同域表现 (训练域 → 同域测试)", same_pairs, cells)
              + "\n" + _metrics_table("跨域零样本泛化", cross_pairs, cells))

    def _ablation_table(cells):
        base_same = cells.get(("KnowledgeLLM", "CAUEEG", "CAUEEG"))
        base_zs = cells.get(("KnowledgeLLM", "CAUEEG", "TUAB"))
        abl = discover_ablations()
        registry = dict(ABLATION_VARIANTS)

        def _auc_mean(cell):
            if not cell:
                return None
            vals = [m.get("auc") for _, m, _ in cell["seeds"]]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        base_mean = {"same": _auc_mean(base_same), "zs": _auc_mean(base_zs)}

        # 定稿在前, 已登记变体按注册顺序, 未登记变体按键名排序兜底
        keys = [""] + [k for k, _ in ABLATION_VARIANTS]
        keys += sorted(k for k in abl if k not in keys)

        def _rel(p):
            try:
                return str(Path(str(p)).relative_to(REPO))
            except ValueError:
                return str(p)

        def _tooltip(label, cell):
            lines = [f"{label} | {cell['mode']} (window AUC)"]
            for seed, met, src in cell["seeds"]:
                v = met.get("auc")
                if v is not None:
                    lines.append(f"seed {seed}: auc={_fmt(v)}")
                lines.append(f"  来源: {_rel(src)}")
            return escape("\n".join(lines))

        rows = []
        best = {"same": None, "zs": None}
        for key in keys:
            if key == "":
                label = "主方法 no_raw (FC+PSD)"
                color = MODEL_COLORS["KnowledgeLLM"]
                same_cell, zs_cell = base_same, base_zs
            else:
                label = registry.get(key, key.upper())
                color = "#64748b"
                rec = abl.get(key, {})
                same_cell = {"mode": "同域", "seeds": [
                    (s, v["same"][0], v["same"][1])
                    for s, v in sorted(rec.items()) if "same" in v]}
                zs_cell = {"mode": "零样本", "seeds": [
                    (s, v["zs"][0], v["zs"][1])
                    for s, v in sorted(rec.items()) if "zs" in v]}
            rows.append((key, label, color, same_cell, zs_cell))
            for kind, cell in (("same", same_cell), ("zs", zs_cell)):
                m = _auc_mean(cell)
                if m is not None and (best[kind] is None or m > best[kind]):
                    best[kind] = m

        thead = ('<th class="left">变体</th>'
                 '<th>同域 CAUEEG AUC</th><th>Δ同域</th>'
                 '<th>→TUAB AUC</th><th>Δ→TUAB</th>')
        tbody = []
        for key, label, color, same_cell, zs_cell in rows:
            tr = [f'<td class="model" style="--mc:{color}">{escape(label)}</td>']
            for kind, cell in (("same", same_cell), ("zs", zs_cell)):
                vals = [m.get("auc") for _, m, _ in cell["seeds"]]
                vals = [v for v in vals if v is not None]
                if not vals:
                    tr.append('<td class="empty" colspan="2">—</td>')
                    continue
                mean = sum(vals) / len(vals)
                style = (' class="best"'
                         if best[kind] is not None
                         and abs(mean - best[kind]) < 1e-9 else "")
                tr.append(
                    f'<td class="cell" title="{_tooltip(label, cell)}">'
                    f'<span{style}>{_cell_text(vals)}</span></td>')
                if key == "" or base_mean[kind] is None:
                    tr.append('<td class="empty">—</td>')
                else:
                    tr.append(
                        f'<td class="cell">{(mean - base_mean[kind]) * 100:+.1f}</td>')
            tbody.append("<tr>" + "".join(tr) + "</tr>")

        return (f'<h2>主方法消融表 (CAUEEG 训练域, window AUC-ROC, 3 seeds)</h2>\n'
                f'<table>\n  <thead>\n    <tr>{thead}</tr>\n  </thead>\n'
                f'  <tbody>\n{chr(10).join(tbody)}\n  </tbody>\n</table>')

    tables = tables + "\n" + _ablation_table(cells)

    n_baseline_cells = 0
    for (model, _, _), c in cells.items():
        if model != "KnowledgeLLM" and len(c["seeds"]) > 0:
            n_baseline_cells += 1

    notes = []
    notes.append("· 口径: <b>KnowledgeLLM = 主方法</b>"
                 "(10s FC+PSD 双模态 [no_raw] + 知识银行 + 冻结 6 层 Llama; "
                 "CAUEEG/TUAB 双训练域, "
                 "<b>3 seeds (42/43/44), 每格 mean ± std</b>); "
                 "<b>EEGNet / EEG-Deformer / LaBraM / NeuroLM / OFA = 10s 段</b>"
                 "(每 60s 窗取首个 10s 段), <b>3 seeds (42/43/44), 每格 mean ± std</b>。")
    notes.append("· 指标口径: 每格报告 <b>Balanced Acc / AUC-ROC / AUC-PR</b> "
                 "三项 (百分数刻度, mean ± std, 单位均为百分点)。"
                 "Balanced Acc = (Sensitivity + Specificity) / 2。")
    notes.append("· <b>所有数值均可溯源</b>: 悬浮提示列出各 seed 数值与来源文件 "
                 "(KnowledgeLLM / EEGNet / OFA / EEG-Deformer 为 metrics JSON 或 "
                 "结果 JSON, NeuroLM 为 test_results.json, LaBraM 为评估日志; "
                 "各项 AUC/AUC-PR 与对应评估日志逐位一致)。")
    notes.append("· 训练设置: EPOCHS=200 / LR=1e-4 / PATIENCE=20 / val-AUC 选模型; "
                 "各基线优化器/调度按各自仓库默认。")
    notes.append("· <b>加粗</b> = 该训练域→目标域该指标的最佳 mean。")
    notes.append("· <b>TUEP 已从主表移除</b>: 所有方法(含基线)在 TUEP 上同域/跨域"
                 "均接近随机(发作间期 EEG 无判别特征), 作为阴性对照单独讨论。")
    notes.append("· 消融表: 主方法 no_raw (FC+PSD) vs 各消融变体 "
                 "(CAUEEG 训练域), 维持 <b>window AUC-ROC</b> 口径 "
                 "(变体未重测新指标), <b>3 seeds mean ± std</b>; "
                 "<b>Δ = 变体 mean − 主方法 mean</b>; <b>加粗</b> = 该列最佳 "
                 "mean; 悬浮提示列出各 seed 数值与来源。A1/A2a/A3c/A3b/A4/A5 "
                 "基于含 raw 的三模态设置运行 (设置差异后续说明)。")

    note_items = "".join(f"<li>{n}</li>" for n in notes)

    # 每个模型的数值来源文件清单 (路径相对 repo 根显示; 非文件条目原样保留)
    def _rel_src(src):
        p = Path(str(src))
        if p.is_absolute():
            try:
                return str(p.relative_to(REPO))
            except ValueError:
                pass
        return str(src)

    model_srcs = {}
    for (model, _, _), c in cells.items():
        for _, _, src in c["seeds"]:
            model_srcs.setdefault(model, set()).add(_rel_src(src))
    src_blocks = []
    for model in MODELS:
        files = sorted(model_srcs.get(model, set()))
        if not files:
            continue
        items = "".join(f"<li><code>{escape(f)}</code></li>" for f in files)
        src_blocks.append(f"<h3>{escape(model)}</h3><ul>{items}</ul>")
    src_section = "\n".join(src_blocks)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>LLM4FC 实验结果对比 (六模型, 3-seed)</title>
<style>
  body {{ font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
          margin: 24px; background: #fafafa; color: #222; }}
  h1 {{ font-size: 20px; margin: 8px 0 4px; }}
  h2 {{ font-size: 15px; margin: 22px 0 8px; }}
  .sub {{ color: #777; font-size: 13px; margin-bottom: 16px; }}
  table {{ border-collapse: collapse; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  th, td {{ border: 1px solid #e3e6eb; padding: 8px 14px; text-align: center; font-size: 14px; }}
  thead th {{ background: #f3f4f8; font-weight: 600; }}
  th.left, td.model, td.src {{ text-align: left; }}
  td.model {{ font-weight: 600; border-left: 4px solid var(--mc); white-space: nowrap; }}
  td.src {{ color: #555; white-space: nowrap; }}
  td.cell span {{ display: block; line-height: 1.45; font-variant-numeric: tabular-nums; }}
  td.cell span.best {{ font-weight: 700; color: #b91c1c; }}
  td.empty {{ color: #bbb; }}
  .notes {{ font-size: 12px; color: #666; margin-top: 14px; line-height: 1.7; max-width: 980px; }}
  .notes li {{ margin: 2px 0; }}
  .srcs {{ font-size: 12px; color: #555; margin-top: 18px; max-width: 980px; }}
  .srcs h3 {{ font-size: 13px; margin: 10px 0 4px; color: #333; }}
  .srcs ul {{ margin: 0 0 4px; padding-left: 18px; line-height: 1.6; }}
  code {{ background: #f0f1f4; padding: 1px 5px; border-radius: 4px; font-size: 12px; }}
</style>
</head>
<body>
<h1>LLM4FC 实验结果对比 (六模型, 3-seed)</h1>
<div class="sub">疾病分类任务 · 同域/跨域分表 · 每格 Balanced Acc / AUC-ROC / AUC-PR (3-seed mean ± std) · 生成于 {now}</div>
{tables}
<div class="notes"><b>说明</b><ul>{note_items}</ul></div>
<div class="srcs"><h2>各模型数值来源文件</h2>{src_section}</div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="六模型 3-seed 结果汇总")
    parser.add_argument("--out", default=str(REPO / "conclusion.html"),
                        help="输出 HTML 路径")
    args = parser.parse_args()

    html = build_html()
    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"HTML 已写入: {out.resolve()}")

    # 终端摘要
    exps = (discover_knowledgellm() + discover_eegnet()
            + discover_deformer() + discover_labram() + discover_neurolm()
            + discover_ofa())
    print(f"共解析 {len(exps)} 条 (含 {len(discover_knowledgellm())} 条 KnowledgeLLM)")
    abl = discover_ablations()
    print(f"消融变体 {len(abl)} 个: "
          + ", ".join(sorted(abl)) if abl else "(无)")


if __name__ == "__main__":
    main()
