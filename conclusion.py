#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conclusion.py
=============
汇总六模型实验结果, 渲染 HTML 对比表格 (默认 conclusion.html)。

口径:
  - TimeLLMKnowledge: win5 主方法 (60s 窗, 5x20s/10s 滑动知识窗), 单 seed 42,
    数值为存档记录 (原始日志已按用户指令清理)。
  - EEGNet / EEG-Deformer / LaBraM / NeuroLM / OFA: 10s 段输入
    (每 60s 窗取首个 10s 段), 3 个 seed (42/43/44), 每格报告 mean ± std。

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
DOMAINS = ["CAUEEG", "TUAB", "TUEP"]
SEEDS = ["42", "43", "44"]

# ─────────────────────────────────────────────────────────────────────────
# 存档: TimeLLMKnowledge win5 主方法 (60s, seed 42) 九格
# 数值来源: 原始 trainer 日志 (10-23 win5 定稿运行), 日志已按用户指令清理,
# 此处为逐格存档 (window 级 test AUC / Acc)。
# ─────────────────────────────────────────────────────────────────────────
WIN5_TLLM = {
    ("CAUEEG", "CAUEEG"): {"auc": 0.8141, "accuracy": 0.7224},
    ("CAUEEG", "TUAB"):   {"auc": 0.7624, "accuracy": 0.6537},
    ("CAUEEG", "TUEP"):   {"auc": 0.5701, "accuracy": 0.5626},
    ("TUAB", "CAUEEG"):   {"auc": 0.6299, "accuracy": 0.6170},
    ("TUAB", "TUAB"):     {"auc": 0.8531, "accuracy": 0.7812},
    ("TUAB", "TUEP"):     {"auc": 0.4869, "accuracy": 0.5088},
    ("TUEP", "CAUEEG"):   {"auc": 0.5055, "accuracy": 0.5051},
    ("TUEP", "TUAB"):     {"auc": 0.5756, "accuracy": 0.5560},
    ("TUEP", "TUEP"):     {"auc": 0.6887, "accuracy": 0.6203},
}


def _tllm_win5_experiments():
    exps = []
    for (src, tgt), m in WIN5_TLLM.items():
        exps.append({
            "model": "TimeLLMKnowledge", "seed": "win5", "source": src,
            "target": tgt, "mode": "同域" if src == tgt else "零样本",
            "metrics": dict(m),
            "src": "archive: win5 主方法 (60s, seed 42) 存档记录",
        })
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
    exps = []
    for seed in SEEDS:
        prefix = "" if seed == "42" else f"_s{seed}"
        # 同域
        for p in sorted(LOG_DIR.glob(f"train_EEGNet{prefix}_Disease*.log")):
            m = re.match(rf"train_EEGNet{prefix}_Disease(\w+)\.log", p.name)
            if not m:
                continue
            dom = m.group(1)
            met = parse_trainer_log(p)
            if met:
                exps.append({
                    "model": "EEGNet", "seed": seed, "source": dom,
                    "target": dom, "mode": "同域", "metrics": met, "src": p,
                })
        # 零样本
        for p in sorted(LOG_DIR.glob(f"test_EEGNet{prefix}_*zero-shot_Disease*.log")):
            m = re.match(rf"test_EEGNet{prefix}_(\w+)_zero-shot_Disease(\w+)\.log",
                         p.name)
            if not m:
                continue
            src = SRC_TAGS.get(m.group(1), m.group(1).upper())
            tgt = m.group(2)
            if src not in DOMAINS or tgt not in DOMAINS:
                continue
            met = parse_trainer_log(p)
            if met:
                exps.append({
                    "model": "EEGNet", "seed": seed, "source": src,
                    "target": tgt, "mode": "零样本", "metrics": met, "src": p,
                })
    return exps


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
            if src == tgt:
                # 同域统一取训练 log.txt (与 s43/s44 口径一致, 避免双源重复)
                continue
            auc = acc = None
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                g = re.search(r"======Global: (\{.*\})", line)
                if g:
                    d = json.loads(g.group(1))
                    if d.get("roc_auc") is not None:
                        auc = float(d["roc_auc"])
                    if d.get("accuracy") is not None:
                        acc = float(d["accuracy"])
            if auc is None:
                continue
            exps.append({
                "model": "LaBraM", "seed": seed, "source": src, "target": tgt,
                "mode": "同域" if src == tgt else "零样本",
                "metrics": {"auc": auc, "accuracy": acc}, "src": p,
            })
    # 同域 (训练 log.txt 最后一行)
    for seed, dirs in LABRAM_TRAIN_DIRS.items():
        for dom, dname in dirs.items():
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
                                "accuracy": float(w["accuracy"])},
                    "src": p,
                })
    return exps


# ─────────────────────────────────────────────────────────────────────────
# 解析: OFA (experiments_llm4fc/*/output.log 的 Test Summary; seed 来自
#       configuration.json; 目录名 Disease{D}_10s_* = 同域评估, ZS_{S}_{T}_*
#       = 零样本)
# ─────────────────────────────────────────────────────────────────────────

OFA_BASE = (REPO / "baseline" / "OneFitsAll" / "Classification"
            / "experiments_llm4fc")
OFA_TEST_RE = re.compile(
    r"Test Summary: loss: ([0-9.eE+-]+) \| accuracy: ([0-9.eE+-]+) \| "
    r"precision: ([0-9.eE+-]+) \| AUROC: ([0-9.eE+-]+)")
OFA_SAME_RE = re.compile(r"Disease(\w+)_10s_\d{4}-\d{2}-\d{2}")
OFA_ZS_RE = re.compile(r"ZS_(\w+)_(\w+)_\d{4}-\d{2}-\d{2}")


def _ofa_seed(d):
    """评估目录的 configuration.json 中 --seed 恒为 42 (脚本未随种子更新),
    故经 load_model 路径回溯到训练目录, 以训练目录的 seed 为准。"""
    cfg = d / "configuration.json"
    if not cfg.exists():
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    lm = str(data.get("load_model", ""))
    m = re.search(r"experiments_llm4fc/(Disease[^/]+)/checkpoints", lm)
    if m:
        train_cfg = (OFA_BASE / m.group(1) / "configuration.json")
        if train_cfg.exists():
            try:
                tdata = json.loads(train_cfg.read_text(encoding="utf-8"))
                return str(tdata.get("seed"))
            except (json.JSONDecodeError, OSError):
                pass
    # 训练目录本身(无 load_model)直接用自身 seed
    return str(data.get("seed"))


def _ofa_test_metrics(log_path):
    for line in log_path.read_text(encoding="utf-8",
                                   errors="replace").splitlines():
        m = OFA_TEST_RE.search(line)
        if m:
            return {"auc": float(m.group(4)),
                    "accuracy": float(m.group(2))}
    return None


def discover_ofa():
    exps = []
    if not OFA_BASE.is_dir():
        return exps
    for d in sorted(OFA_BASE.iterdir()):
        if not d.is_dir():
            continue
        seed = _ofa_seed(d)
        if seed not in SEEDS:
            continue
        log = d / "output.log"
        if not log.exists():
            continue
        met = _ofa_test_metrics(log)
        if met is None:
            continue
        m_same = OFA_SAME_RE.match(d.name)
        if m_same:
            dom = m_same.group(1)
            exps.append({
                "model": "OFA", "seed": seed, "source": dom, "target": dom,
                "mode": "同域", "metrics": met, "src": log,
            })
            continue
        m_zs = OFA_ZS_RE.match(d.name)
        if m_zs:
            src, tgt = m_zs.groups()
            if src in DOMAINS and tgt in DOMAINS:
                exps.append({
                    "model": "OFA", "seed": seed, "source": src,
                    "target": tgt, "mode": "零样本", "metrics": met,
                    "src": log,
                })
    return exps


# ─────────────────────────────────────────────────────────────────────────
# 聚合 + 渲染
# ─────────────────────────────────────────────────────────────────────────

MODELS = ["EEGNet", "TimeLLMKnowledge", "NeuroLM", "LaBraM", "OFA",
          "EEG-Deformer"]
MODEL_COLORS = {
    "EEGNet": "#2563eb", "TimeLLMKnowledge": "#7c3aed", "NeuroLM": "#059669",
    "LaBraM": "#b45309", "OFA": "#db2777", "EEG-Deformer": "#0891b2",
}


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if x is not None else "—"


def _cell_text(vals):
    if len(vals) == 1:
        return _fmt(vals[0])
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    std = var ** 0.5
    return f"{_fmt(mean)} ± {_fmt(std, 3)}"


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
            lines.append(f"mean ± std: {_fmt(mean)} ± {_fmt(var ** 0.5, 3)}")
    return escape("\n".join(lines))


def build_html():
    # 收集
    exps = (_tllm_win5_experiments() + discover_eegnet()
            + discover_deformer() + discover_labram() + discover_neurolm()
            + discover_ofa())

    # 聚合: (model, source, target) -> {mode, seeds: [(seed, metrics, src)]}
    cells = {}
    for e in exps:
        key = (e["model"], e["source"], e["target"])
        c = cells.setdefault(key, {"mode": e["mode"], "seeds": []})
        c["seeds"].append((e["seed"], e["metrics"], e["src"]))

    def _source_table(source):
        rows = []
        for model in MODELS:
            row = {}
            for d in DOMAINS:
                row[d] = cells.get((model, source, d))
            rows.append((model, row))

        best = {d: {"auc": None, "accuracy": None} for d in DOMAINS}
        for _, row in rows:
            for d in DOMAINS:
                c = row[d]
                if not c:
                    continue
                for key in ("auc", "accuracy"):
                    vals = [m.get(key) for _, m, _ in c["seeds"]]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        mean = sum(vals) / len(vals)
                        cur = best[d][key]
                        if cur is None or mean > cur:
                            best[d][key] = mean

        thead1 = ['<th rowspan="2" class="left">模型</th>']
        thead2 = []
        for d in DOMAINS:
            thead1.append(f'<th colspan="2">{escape(d)}</th>')
            thead2.append("<th>AUC</th><th>Acc</th>")

        tbody = []
        for model, row in rows:
            color = MODEL_COLORS.get(model, "#666")
            tr = [f'<td class="model" style="--mc:{color}">{escape(model)}</td>']
            for d in DOMAINS:
                c = row[d]
                if c:
                    for key in ("auc", "accuracy"):
                        vals = [m.get(key) for _, m, _ in c["seeds"]]
                        vals = [v for v in vals if v is not None]
                        if not vals:
                            tr.append('<td class="empty">—</td>')
                            continue
                        mean = sum(vals) / len(vals)
                        style = (' class="best"'
                                 if best[d][key] is not None
                                 and abs(mean - best[d][key]) < 1e-9 else "")
                        tr.append(
                            f'<td class="cell" title="{_cell_tooltip(model, source, d, c, key)}">'
                            f'<span{style}>{_cell_text(vals)}</span></td>')
                else:
                    tr.append('<td class="empty" colspan="2">—</td>')
            tbody.append("<tr>" + "".join(tr) + "</tr>")

        return (f'<h2>训练域 = {escape(source)}</h2>\n'
                f'<table>\n  <thead>\n    <tr>{"".join(thead1)}</tr>\n'
                f'    <tr>{"".join(thead2)}</tr>\n  </thead>\n'
                f'  <tbody>\n{chr(10).join(tbody)}\n  </tbody>\n</table>')

    tables = "\n".join(_source_table(s) for s in DOMAINS)

    n_baseline_cells = 0
    for (model, _, _), c in cells.items():
        if model != "TimeLLMKnowledge" and len(c["seeds"]) > 0:
            n_baseline_cells += 1

    notes = []
    notes.append("· 口径: <b>TimeLLMKnowledge = win5 主方法</b>"
                 "(60s 窗, 5×20s/10s 滑动知识窗, seed 42, 存档数值); "
                 "<b>EEGNet / EEG-Deformer / LaBraM / NeuroLM / OFA = 10s 段</b>"
                 "(每 60s 窗取首个 10s 段), <b>3 seeds (42/43/44), 每格 mean ± std</b>。")
    notes.append("· 训练设置: EPOCHS=200 / LR=1e-4 / PATIENCE=20 / val-AUC 选模型; "
                 "各基线优化器/调度按各自仓库默认。")
    notes.append("· <b>加粗</b> = 该测试域该指标的最佳 mean; 悬浮提示列出各 seed 数值与来源。")

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
<div class="sub">疾病分类任务 · 按训练域分表 · 每格 AUC / Acc (baseline: mean ± std; TLLM: win5 单 seed) · 生成于 {now}</div>
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
    exps = (_tllm_win5_experiments() + discover_eegnet()
            + discover_deformer() + discover_labram() + discover_neurolm()
            + discover_ofa())
    print(f"共解析 {len(exps)} 条 (含 {len(_tllm_win5_experiments())} 条 TLLM 存档)")


if __name__ == "__main__":
    main()
