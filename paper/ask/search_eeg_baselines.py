#!/usr/bin/env python3
"""Discover traditional EEG classifiers, LLM-based methods, and EEG foundation models via OpenAlex.

OpenAlex works without an API key for light use. If a key is available, set
OPENALEX_API_KEY in the environment; this script never stores credentials.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


QUERY_GROUPS = {
    "traditional_small_eeg_classification": [
        'raw EEG classification CNN transformer',
        'EEGNet EEG classification',
        'ShallowConvNet DeepConvNet EEG classification',
        'EEG Conformer classification',
        'EEG-Deformer classification',
        'LMDA-Net EEG classification',
        'HiSAM-MambaNet EEG classification',
    ],
    "llm_based_eeg_classification": [
        'EEG large language model classification',
        'EEG LLM classification fine tuning',
        'EEG modality reprogramming frozen LLM classification',
        'EEG-GPT classification interpretation',
        'Tokenized EEG signals large language models epilepsy detection Information Fusion',
        'EEG Agent unified framework automated EEG analysis AAAI',
        'Time-LLM time series classification reprogramming',
        'One Fits All pretrained language model time series classification',
    ],
    "eeg_foundation_models": [
        'EEG foundation model',
        'EEG pretrained transformer classification',
        'LaBraM EEG',
        'NeuroLM EEG',
        'CBraMod EEG',
        'EEGPT EEG',
        'BIOT biosignal transformer EEG',
        'EEG2Rep EEG representation learning',
        'FEMBA EEG foundation model Mamba',
        'Gram large general EEG model raw data classification restoration',
        'EEGMamba EEG foundation model Mamba',
        'Neuro-GPT foundation model EEG',
    ],
}


def compact(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def reconstruct_abstract(work: dict[str, Any]) -> str:
    inverted = work.get("abstract_inverted_index") or {}
    positions = sorted(
        (index, word)
        for word, indexes in inverted.items()
        for index in indexes
    )
    return compact(" ".join(word for _, word in positions))


def get_works(query: str, per_query: int, api_key: str | None) -> dict[str, Any]:
    params = {
        "search": query,
        "per-page": str(per_query),
        "sort": "relevance_score:desc",
        "select": ",".join(
            [
                "id",
                "doi",
                "title",
                "publication_year",
                "type",
                "cited_by_count",
                "authorships",
                "primary_location",
                "open_access",
                "abstract_inverted_index",
            ]
        ),
    }
    if api_key:
        params["api_key"] = api_key
    request = Request(
        "https://api.openalex.org/works?" + urlencode(params),
        headers={"User-Agent": "LLM4FC-EEG-literature-survey/1.0"},
    )
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if "results" not in payload:
        raise RuntimeError(f"OpenAlex response has no results for {query!r}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-query", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    if args.per_query < 1:
        parser.error("--per-query must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("OPENALEX_API_KEY")
    raw_by_query: dict[str, Any] = {}
    works_by_id: dict[str, dict[str, Any]] = {}
    groups_by_id: defaultdict[str, set[str]] = defaultdict(set)

    for group, queries in QUERY_GROUPS.items():
        for query in queries:
            print(f"Querying {group}: {query}")
            try:
                payload = get_works(query, args.per_query, api_key)
            except Exception as exc:
                print(f"  failed: {exc}")
                continue
            raw_by_query[query] = payload
            for work in payload.get("results", []):
                work_id = work.get("id")
                if work_id:
                    works_by_id[work_id] = work
                    groups_by_id[work_id].add(group)
            time.sleep(max(args.sleep, 0.0))

    rows = []
    for work_id, work in works_by_id.items():
        location = work.get("primary_location") or {}
        source = location.get("source") or {}
        rows.append(
            {
                "id": work_id,
                "title": compact(work.get("title")),
                "year": work.get("publication_year"),
                "type": work.get("type", ""),
                "venue": source.get("display_name", ""),
                "cited_by_count": work.get("cited_by_count", 0),
                "doi": work.get("doi") or "",
                "url": location.get("landing_page_url") or work.get("doi") or work_id,
                "groups": ";".join(sorted(groups_by_id[work_id])),
                "abstract": reconstruct_abstract(work),
            }
        )
    rows.sort(key=lambda row: (row["year"] or 0, row["cited_by_count"]), reverse=True)

    (output_dir / "openalex_eeg_raw.json").write_text(
        json.dumps(raw_by_query, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    (output_dir / "openalex_eeg_candidates.json").write_text(
        json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    fields = [
        "id",
        "title",
        "year",
        "type",
        "venue",
        "cited_by_count",
        "doi",
        "url",
        "groups",
        "abstract",
    ]
    with (output_dir / "openalex_eeg_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} unique works to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
