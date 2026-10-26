#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conclusion.py
=============
汇总 EEGNet / TimeLLMKnowledge / NeuroLM 现有的全部实验结果,
渲染为【一个】HTML 对比表格 (默认写回本目录下的 conclusion.html)。

数据来源(全部从磁盘现有文件解析, 不重新训练):
  1. log_dir/train_{MODEL}_Disease{DOMAIN}.log              -> 同域测试 (best-epoch 指标块)
  2. log_dir/test_{MODEL}[_tuab|_tuep]_zero-shot_Disease{T}.log -> 零样本跨域测试
  3. baseline/NeuroLM/out_llm4fc/test_results.json          -> NeuroLM (CAUEEG 微调, 评估三域)
  4. baseline/NeuroLM/out_llm4fc_tuab/test_results.json     -> NeuroLM (TUAB 微调, 评估三域)
  5. baseline/NeuroLM/out_llm4fc_tuep/test_results.json     -> NeuroLM (TUEP 微调, 评估三域)

输出: 按训练域分三张对比表 (训练域 = CAUEEG / TUAB / TUEP)。

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
NEUROLM_OUT = REPO / "baseline" / "NeuroLM"

DOMAINS = ["CAUEEG", "TUAB", "TUEP"]  # 测试域列顺序

# ---------------------------------------------------------------------------
# 解析: EEGNet / TimeLLMKnowledge 的 trainer 日志
# ---------------------------------------------------------------------------

# 日志指标行形如: 10/15/2026 21:21:07 - INFO - utils.trainer -   AUC: 0.83338
# (普通评估块缩进 3 空格, "Best epoch test result" 块缩进 5 空格, 统一按 \s+ 匹配)
METRIC_LINE_RE = re.compile(
    r"^\S+ \S+ - INFO - \S+ -\s+(Loss|AUC|Accuracy|Precision|Specificity|Sensitivity): ([0-9.eE+-]+)"
)
PRETRAIN_PATH_RE = re.compile(r"^\s*pretrain_path: (\S+)", re.MULTILINE)


def parse_trainer_log(path):
    """提取日志中最后一个完整指标块 (train 日志即 best-epoch test 结果,
    test 日志即最终评估结果)。返回 dict, 失败返回 None。"""
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


def parse_pretrain_path(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    m = PRETRAIN_PATH_RE.search(text)
    return m.group(1) if m else None


def discover_trainer_experiments():
    """扫描 log_dir, 收集 EEGNet / TimeLLMKnowledge 的同域 + 零样本结果。"""
    exps = []
    # 同域: train_{MODEL}_Disease{DOMAIN}.log 的最后一个指标块
    for p in sorted(LOG_DIR.glob("train_*_Disease*.log")):
        m = re.match(r"train_(\w+)_Disease(\w+)\.log", p.name)
        if not m:
            continue
        model, domain = m.groups()
        metrics = parse_trainer_log(p)
        if not metrics:
            print(f"[warn] 未在 {p} 中找到指标块, 跳过", file=sys.stderr)
            continue
        exps.append({
            "model": model, "source": domain, "target": domain,
            "mode": "同域", "metrics": metrics, "src": p,
            "checkpoint": None,
        })
    # 零样本: test_{MODEL}[_tuab|_tuep]_zero-shot_Disease{TARGET}.log
    for p in sorted(LOG_DIR.glob("test_*zero-shot_Disease*.log")):
        m = re.match(r"test_(\w+?)(_tuab|_tuep)?_zero-shot_Disease(\w+)\.log",
                     p.name)
        if not m:
            continue
        model, suffix, target = m.groups()
        metrics = parse_trainer_log(p)
        if not metrics:
            print(f"[warn] 未在 {p} 中找到指标块, 跳过", file=sys.stderr)
            continue
        exps.append({
            "model": model,
            "source": {"_tuab": "TUAB", "_tuep": "TUEP"}.get(suffix, "CAUEEG"),
            "target": target,
            "mode": "零样本",
            "metrics": metrics,
            "src": p,
            "checkpoint": parse_pretrain_path(p),
        })
    return exps


# ---------------------------------------------------------------------------
# 解析: NeuroLM 结果 (JSON)
# ---------------------------------------------------------------------------


def _neurolm_window(data):
    """从 JSON 片段里取 window 级指标, 返回 (auc, pr_auc, acc, bacc, f1)。"""
    w = data["window"]
    return (w["roc_auc"], w["pr_auc"], w["accuracy"],
            w["balanced_accuracy"], w["f1"])


def discover_neurolm_experiments():
    exps = []
    # 微调模型: out_llm4fc (CAUEEG 训练) / out_llm4fc_tuab (TUAB 训练) /
    #          out_llm4fc_tuep (TUEP 训练)
    for sub, source in [("out_llm4fc", "CAUEEG"), ("out_llm4fc_tuab", "TUAB"),
                        ("out_llm4fc_tuep", "TUEP")]:
        p = NEUROLM_OUT / sub / "test_results.json"
        if not p.exists():
            print(f"[warn] 未找到 {p}, 跳过", file=sys.stderr)
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for dom, dom_data in data.items():
            if dom not in DOMAINS or not isinstance(dom_data, dict):
                continue
            test_w = _neurolm_window(dom_data["test"])
            val_w = _neurolm_window(dom_data["val"])
            test_s = dom_data["test"].get("subject")
            val_s = dom_data["val"].get("subject")
            metrics = {
                "auc": test_w[0], "accuracy": test_w[2],
                "neurolm_window_test": test_w,
                "neurolm_window_val": val_w,
                "neurolm_subject_test": test_s,
                "neurolm_subject_val": val_s,
            }
            exps.append({
                "model": "NeuroLM", "source": source, "target": dom,
                "mode": "同域" if dom == source else "零样本",
                "metrics": metrics, "src": p, "checkpoint": None,
            })
    return exps


# ---------------------------------------------------------------------------
# 解析: LaBraM 结果 (持久评估日志 baseline/LaBraM/log_llm4fc/<src>__<target>.log)
# ---------------------------------------------------------------------------

LABRAM_LOG_DIR = REPO / "baseline" / "LaBraM" / "log_llm4fc"
LABRAM_SRC_MAP = {"out_llm4fc_caueeg": "CAUEEG",
                  "out_llm4fc_tuab": "TUAB",
                  "out_llm4fc_tuep": "TUEP"}
LABRAM_NAME_RE = re.compile(r"(out_llm4fc_\w+)__Disease(\w+)\.log")


def discover_labram_experiments():
    """从 LaBraM --eval 日志提取最终测试指标 (最后一行的整体统计)。"""
    exps = []
    if not LABRAM_LOG_DIR.is_dir():
        print(f"[warn] 未找到 {LABRAM_LOG_DIR}, 跳过 LaBraM", file=sys.stderr)
        return exps
    for p in sorted(LABRAM_LOG_DIR.glob("*.log")):
        m = LABRAM_NAME_RE.match(p.name)
        if not m:
            continue
        source = LABRAM_SRC_MAP.get(m.group(1))
        target = m.group(2)
        if source is None or target not in DOMAINS:
            continue
        metrics = {}
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            # 全局真值行（metric_logger global_avg；显示行括号值只是短窗滑动平均）
            m = re.search(r"======Global: (\{.*\})", line)
            if not m:
                continue
            d = json.loads(m.group(1))
            if d.get("roc_auc") is not None:
                metrics["auc"] = float(d["roc_auc"])
            if d.get("accuracy") is not None:
                metrics["accuracy"] = float(d["accuracy"])
        if "auc" not in metrics:
            print(f"[warn] 未在 {p} 中找到指标, 跳过", file=sys.stderr)
            continue
        exps.append({
            "model": "LaBraM", "source": source, "target": target,
            "mode": "同域" if source == target else "零样本",
            "metrics": metrics, "src": p, "checkpoint": None,
        })
    return exps


# ---------------------------------------------------------------------------
# HTML 渲染
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "EEGNet": "#2563eb",
    "TimeLLMKnowledge": "#7c3aed",
    "NeuroLM": "#059669",
    "LaBraM": "#b45309",
}


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if x is not None else "—"


def _cell_tooltip(exp):
    """单格悬浮提示: 来源文件 + 全部可用指标。"""
    lines = [f"{exp['model']} | 训练域 {exp['source']} → {exp['target']} ({exp['mode']})",
             f"来源: {exp['src']}"]
    m = exp["metrics"]
    if exp["checkpoint"]:
        lines.append(f"checkpoint: {exp['checkpoint']}")
    if "neurolm_window_test" in m:
        auc, pr, acc, bacc, f1 = m["neurolm_window_test"]
        lines.append(f"window: AUC={_fmt(auc)}  PR-AUC={_fmt(pr)}  Acc={_fmt(acc)}  "
                     f"BAcc={_fmt(bacc)}  F1={_fmt(f1)}")
        if m["neurolm_window_val"] is not None:
            auc, pr, acc, bacc, f1 = m["neurolm_window_val"]
            lines.append(f"window(val): AUC={_fmt(auc)}  Acc={_fmt(acc)}")
        for tag, key in (("subject", "neurolm_subject_test"), ("subject(val)", "neurolm_subject_val")):
            s = m[key]
            if s:
                lines.append(f"{tag}: AUC={_fmt(s.get('roc_auc'))}  "
                             f"Acc={_fmt(s.get('accuracy'))}  F1={_fmt(s.get('f1'))}")
    else:
        for k in ("loss", "auc", "accuracy", "precision", "specificity", "sensitivity"):
            if k in m:
                lines.append(f"{k.capitalize()}: {_fmt(m[k])}")
    return escape("\n".join(lines))


def _metric_td(exp, key, best_val):
    """一个指标一格 (AUC 或 Acc), 加粗标记该列最佳。"""
    m = exp["metrics"]
    v = m.get(key)
    style = ' class="best"' if best_val is not None and v == best_val else ""
    return (f'<td class="cell" title="{_cell_tooltip(exp)}">'
            f'<span{style}>{_fmt(v)}</span></td>')


MODELS = ["EEGNet", "TimeLLMKnowledge", "NeuroLM", "LaBraM"]


def _source_table(exps, source):
    """一张训练域表: 行 = 模型, 列 = 测试域 × (AUC, Acc)。"""
    rows = []  # (model, cells)
    for model in MODELS:
        cells = {d: None for d in DOMAINS}
        for e in exps:
            if e["model"] == model and e["source"] == source:
                cells[e["target"]] = e
        rows.append((model, cells))

    # 每列(测试域 × 指标)的最佳值, 用于加粗
    best = {d: {"auc": None, "accuracy": None} for d in DOMAINS}
    for _, cells in rows:
        for d in DOMAINS:
            e = cells[d]
            if not e:
                continue
            m = e["metrics"]
            for key in ("auc", "accuracy"):
                v = m.get(key)
                cur = best[d][key]
                if v is not None and (cur is None or v > cur):
                    best[d][key] = v

    # 表头: 模型 (rowspan=2) + 每个测试域 AUC/Acc 两列
    thead1 = ['<th rowspan="2" class="left">模型</th>']
    thead2 = []
    for d in DOMAINS:
        thead1.append(f'<th colspan="2">{escape(d)}</th>')
        thead2.append("<th>AUC</th><th>Acc</th>")

    tbody = []
    for model, cells in rows:
        color = MODEL_COLORS.get(model, "#666")
        tr = [f'<td class="model" style="--mc:{color}">{escape(model)}</td>']
        for d in DOMAINS:
            e = cells[d]
            if e:
                tr.append(_metric_td(e, "auc", best[d]["auc"]))
                tr.append(_metric_td(e, "accuracy", best[d]["accuracy"]))
            else:
                tr.append('<td class="empty" colspan="2">—</td>')
        tbody.append("<tr>" + "".join(tr) + "</tr>")

    return (f'<h2>训练域 = {escape(source)}</h2>\n'
            f'<table>\n  <thead>\n    <tr>{"".join(thead1)}</tr>\n'
            f'    <tr>{"".join(thead2)}</tr>\n  </thead>\n'
            f'  <tbody>\n{chr(10).join(tbody)}\n  </tbody>\n</table>')


def build_html(exps):
    tables = "\n".join(_source_table(exps, s) for s in ("CAUEEG", "TUAB", "TUEP"))

    sources = sorted({str(e["src"]) for e in exps})
    notes = []
    notes.append("· EEGNet / TimeLLMKnowledge 指标来自 trainer 日志(测试集, best-epoch 模型, 60s 窗输入); "
                 "NeuroLM / LaBraM 指标为 window 级(测试集, 每 60s 窗取首个 10s 段输入, "
                 "lr 1e-4 / 200 epochs / patience 20 / val-AUC 选模型), subject 级见悬浮提示。")
    notes.append("· <b>加粗</b> = 该测试域该指标的最佳值; 各格悬浮提示标注同域/零样本及完整指标。")

    src_items = "".join(f"<li><code>{escape(s)}</code></li>" for s in sources)
    note_items = "".join(f"<li>{n}</li>" for n in notes)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>LLM4FC 实验结果对比 (EEGNet / TimeLLMKnowledge / NeuroLM)</title>
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
  .notes, .srcs {{ font-size: 12px; color: #666; margin-top: 14px; line-height: 1.7; max-width: 980px; }}
  .notes li, .srcs li {{ margin: 2px 0; }}
  code {{ background: #f0f1f4; padding: 1px 5px; border-radius: 4px; font-size: 12px; }}
</style>
</head>
<body>
<h1>LLM4FC 实验结果对比 (EEGNet / TimeLLMKnowledge / NeuroLM)</h1>
<div class="sub">疾病分类任务 · 按训练域分表 · 每个测试域两列 (AUC / Acc) · 生成于 {now}</div>
{tables}
<div class="notes"><b>说明</b><ul>{note_items}</ul></div>
<div class="srcs"><b>数据来源</b><ul>{src_items}</ul></div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="汇总 EEGNet/TimeLLMKnowledge/NeuroLM 现有结果, 生成 HTML 对比表格")
    parser.add_argument("--out", default=str(REPO / "conclusion.html"),
                        help="输出 HTML 路径 (默认: 本脚本所在目录下 conclusion.html)")
    args = parser.parse_args()

    exps = (discover_trainer_experiments() + discover_neurolm_experiments()
            + discover_labram_experiments())
    if not exps:
        print("未解析到任何实验结果, 请确认 log_dir 与 baseline/NeuroLM/out_* 文件存在。",
              file=sys.stderr)
        sys.exit(1)

    html = build_html(exps)
    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"共汇总 {len(exps)} 条实验结果, HTML 已写入: {out.resolve()}")

    # 终端摘要
    print(f"\n{'模型':<22}{'训练域':<10}{'→ 测试域':<10}{'AUC':>9}  {'Acc':>9}")
    print("-" * 64)
    for e in sorted(exps, key=lambda x: (x["model"], x["source"], x["target"])):
        m = e["metrics"]
        print(f"{e['model']:<22}{e['source']:<10}{e['target']:<12}"
              f"{_fmt(m.get('auc')):>9}  {_fmt(m.get('accuracy')):>9}")


if __name__ == "__main__":
    main()
