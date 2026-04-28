#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis_utils import bootstrap_macro_average, bootstrap_macro_delta


RAG_DATASETS = ("nq", "triviaqa", "hotpotqa", "popqa")
RECALL_DATASETS = ("json_kv", "ruler_niah_mk_2", "ruler_niah_mk_3", "ruler_niah_mv")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize COS484 HELMET runs")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag_substring", default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--position_length", type=int, default=32768)
    return parser.parse_args()


def load_run(path: str) -> Optional[Dict[str, Any]]:
    if path.endswith(".score") or path.endswith(".batch") or path.endswith(".result"):
        return None
    with open(path) as f:
        payload = json.load(f)
    rows = payload.get("data", [])
    if not rows:
        return None
    dataset = rows[0].get("dataset") or payload.get("args", {}).get("datasets")
    method = payload.get("method") or rows[0].get("method", "unknown")
    length = payload.get("args", {}).get("input_max_length")
    metric_name = rows[0].get("primary_metric_name") or _infer_primary_metric(dataset, payload)
    return {
        "path": path,
        "dataset": dataset,
        "method": method,
        "length": length,
        "metric_name": metric_name,
        "payload": payload,
        "rows": rows,
    }


def _infer_primary_metric(dataset: str, payload: Dict[str, Any]) -> Optional[str]:
    metrics = payload.get("averaged_metrics", {})
    if dataset and any(name in dataset for name in RAG_DATASETS) and "substring_exact_match" in metrics:
        return "substring_exact_match"
    for candidate in ("substring_exact_match", "exact_match", "ruler_recall"):
        if candidate in metrics:
            return candidate
    return next(iter(metrics), None)


def iter_runs(input_dir: str, tag_substring: Optional[str]) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        if tag_substring and tag_substring not in os.path.basename(path):
            continue
        run = load_run(path)
        if run is not None:
            runs.append(run)
    return runs


def is_rag_dataset(dataset: str) -> bool:
    return any(name in dataset for name in RAG_DATASETS)


def is_recall_dataset(dataset: str) -> bool:
    return any(name in dataset for name in RECALL_DATASETS)


def get_scores(run: Dict[str, Any]) -> List[float]:
    metric = run["metric_name"]
    scores = []
    for row in run["rows"]:
        score = row.get("primary_score")
        if score is None and metric is not None:
            score = row.get(metric)
        if score is not None:
            scores.append(float(score))
    return scores


def summarize_macro(runs: List[Dict[str, Any]], dataset_filter, bootstrap_samples: int, seed: int) -> List[Dict[str, Any]]:
    grouped = defaultdict(dict)
    for run in runs:
        if not dataset_filter(run["dataset"]):
            continue
        grouped[(run["method"], run["length"])][run["dataset"]] = get_scores(run)

    summary = []
    for (method, length), dataset_scores in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        if not dataset_scores:
            continue
        stats = bootstrap_macro_average(dataset_scores, num_bootstrap_samples=bootstrap_samples, seed=seed)
        summary.append({
            "method": method,
            "length": length,
            **stats,
        })
    return summary


def summarize_deltas(runs: List[Dict[str, Any]], bootstrap_samples: int, seed: int) -> List[Dict[str, Any]]:
    grouped = defaultdict(dict)
    for run in runs:
        if not is_rag_dataset(run["dataset"]):
            continue
        grouped[(run["method"], run["length"])][run["dataset"]] = get_scores(run)

    summary = []
    for (method, length), dataset_scores in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        if method == "vanilla":
            continue
        vanilla_scores = grouped.get(("vanilla", length))
        if not vanilla_scores:
            continue
        stats = bootstrap_macro_delta(
            vanilla_scores,
            dataset_scores,
            num_bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        summary.append({
            "method": method,
            "length": length,
            **stats,
        })
    return summary


def summarize_position(runs: List[Dict[str, Any]], target_length: int) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for run in runs:
        if not is_rag_dataset(run["dataset"]) or run["length"] != target_length:
            continue
        for row in run["rows"]:
            bucket = row.get("answer_position_bucket")
            score = row.get("primary_score")
            if bucket is None or score is None:
                continue
            grouped[(run["method"], bucket)].append(float(score))
    summary = []
    for (method, bucket), scores in sorted(grouped.items()):
        summary.append({
            "method": method,
            "bucket": bucket,
            "mean_score": sum(scores) / len(scores),
            "count": len(scores),
        })
    return summary


def summarize_window_slices(runs: List[Dict[str, Any]], target_length: int) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for run in runs:
        if not is_rag_dataset(run["dataset"]) or run["length"] != target_length:
            continue
        for row in run["rows"]:
            score = row.get("primary_score")
            if score is None:
                continue
            for key, value in row.items():
                if not key.startswith("gold_inside_window_"):
                    continue
                grouped[(run["method"], key, bool(value))].append(float(score))
    summary = []
    for (method, window_key, inside), scores in sorted(grouped.items()):
        summary.append({
            "method": method,
            "window": window_key.replace("gold_inside_window_", ""),
            "inside_window": inside,
            "mean_score": sum(scores) / len(scores),
            "count": len(scores),
        })
    return summary


def summarize_diagnostic(runs: List[Dict[str, Any]], target_length: int) -> Dict[str, Any]:
    failures = defaultdict(int)
    totals = defaultdict(int)
    accuracy_by_bucket = defaultdict(list)
    for run in runs:
        if run["method"] != "vanilla" or run["length"] != target_length or not is_rag_dataset(run["dataset"]):
            continue
        for row in run["rows"]:
            score = float(row.get("primary_score", 0.0))
            bucket = row.get("answer_position_bucket")
            if bucket is not None:
                accuracy_by_bucket[bucket].append(score)
            if score >= 1.0:
                continue
            for key, value in row.items():
                if not key.startswith("gold_inside_window_"):
                    continue
                window = key.replace("gold_inside_window_", "")
                totals[window] += 1
                if not value:
                    failures[window] += 1
    return {
        "accuracy_by_bucket": {
            bucket: sum(scores) / len(scores) for bucket, scores in accuracy_by_bucket.items()
        },
        "failure_fraction_outside_window": {
            window: (failures[window] / totals[window]) if totals[window] else None
            for window in sorted(totals)
        },
    }


def write_json(path: str, payload: Any):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_make_plots(output_dir: str, rag_summary, recall_summary, delta_summary, position_summary, window_summary):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    def _group(rows, x_key, y_key):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["method"]].append((row[x_key], row[y_key]))
        return grouped

    if rag_summary or recall_summary:
        plt.figure(figsize=(7, 4))
        for label, rows in (("RAG", rag_summary), ("Recall", recall_summary)):
            rows = sorted(rows, key=lambda row: row["length"])
            if not rows:
                continue
            plt.plot([row["length"] for row in rows], [row["point_estimate"] for row in rows], marker="o", label=label)
        plt.xlabel("Input Length")
        plt.ylabel("Macro Accuracy")
        plt.title("Vanilla Recall vs RAG")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "vanilla_recall_vs_rag.png"))
        plt.close()

    if delta_summary:
        plt.figure(figsize=(8, 4))
        for method, points in sorted(_group(delta_summary, "length", "point_estimate").items()):
            points = sorted(points)
            plt.plot([x for x, _ in points], [y for _, y in points], marker="o", label=method)
        plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
        plt.xlabel("Input Length")
        plt.ylabel("Macro Delta vs Vanilla")
        plt.title("RAG Method Delta vs Vanilla")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rag_delta_vs_vanilla.png"))
        plt.close()

    if position_summary:
        ordered_buckets = ["early", "middle", "late"]
        plt.figure(figsize=(8, 4))
        methods = sorted({row["method"] for row in position_summary})
        for method in methods:
            rows = {row["bucket"]: row["mean_score"] for row in position_summary if row["method"] == method}
            plt.plot(ordered_buckets, [rows.get(bucket, 0.0) for bucket in ordered_buckets], marker="o", label=method)
        plt.xlabel("Answer Position Bucket")
        plt.ylabel("Mean Accuracy")
        plt.title("RAG Accuracy by Evidence Position")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rag_accuracy_by_position.png"))
        plt.close()

    if window_summary:
        for window in sorted({row["window"] for row in window_summary}):
            plt.figure(figsize=(8, 4))
            methods = sorted({row["method"] for row in window_summary if row["window"] == window})
            for method in methods:
                rows = {
                    "inside": row["mean_score"]
                    for row in window_summary
                    if row["method"] == method and row["window"] == window and row["inside_window"]
                }
                rows["outside"] = next(
                    (
                        row["mean_score"]
                        for row in window_summary
                        if row["method"] == method and row["window"] == window and not row["inside_window"]
                    ),
                    0.0,
                )
                plt.plot(["outside", "inside"], [rows.get("outside", 0.0), rows.get("inside", 0.0)], marker="o", label=method)
            plt.xlabel("Gold Evidence in Trailing Window")
            plt.ylabel("Mean Accuracy")
            plt.title(f"RAG Accuracy by Window Membership ({window} tokens)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"rag_accuracy_window_{window}.png"))
            plt.close()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    runs = iter_runs(args.input_dir, args.tag_substring)

    rag_summary = summarize_macro(runs, is_rag_dataset, args.bootstrap_samples, args.seed)
    recall_summary = [row for row in summarize_macro(runs, is_recall_dataset, args.bootstrap_samples, args.seed) if row["method"] == "vanilla"]
    delta_summary = summarize_deltas(runs, args.bootstrap_samples, args.seed)
    position_summary = summarize_position(runs, args.position_length)
    window_summary = summarize_window_slices(runs, args.position_length)
    diagnostic_summary = summarize_diagnostic(runs, args.position_length)

    write_json(os.path.join(args.output_dir, "rag_macro_summary.json"), rag_summary)
    write_json(os.path.join(args.output_dir, "recall_macro_summary.json"), recall_summary)
    write_json(os.path.join(args.output_dir, "rag_delta_summary.json"), delta_summary)
    write_json(os.path.join(args.output_dir, "rag_position_summary.json"), position_summary)
    write_json(os.path.join(args.output_dir, "rag_window_summary.json"), window_summary)
    write_json(os.path.join(args.output_dir, "rag_32k_diagnostic.json"), diagnostic_summary)

    write_csv(os.path.join(args.output_dir, "rag_macro_summary.csv"), rag_summary)
    write_csv(os.path.join(args.output_dir, "recall_macro_summary.csv"), recall_summary)
    write_csv(os.path.join(args.output_dir, "rag_delta_summary.csv"), delta_summary)
    write_csv(os.path.join(args.output_dir, "rag_position_summary.csv"), position_summary)
    write_csv(os.path.join(args.output_dir, "rag_window_summary.csv"), window_summary)

    maybe_make_plots(args.output_dir, rag_summary, recall_summary, delta_summary, position_summary, window_summary)
