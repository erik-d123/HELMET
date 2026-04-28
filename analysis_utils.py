from __future__ import annotations

import random
from typing import Dict, Iterable, List


def macro_average(scores_by_dataset: Dict[str, Iterable[float]]) -> float:
    dataset_means = []
    for scores in scores_by_dataset.values():
        scores = list(scores)
        if not scores:
            raise ValueError("Each dataset must contain at least one score")
        dataset_means.append(sum(scores) / len(scores))
    if not dataset_means:
        raise ValueError("scores_by_dataset must not be empty")
    return sum(dataset_means) / len(dataset_means)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    weight = pos - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def bootstrap_macro_average(
    scores_by_dataset: Dict[str, Iterable[float]],
    num_bootstrap_samples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    normalized = {dataset: list(scores) for dataset, scores in scores_by_dataset.items()}
    point_estimate = macro_average(normalized)
    rng = random.Random(seed)
    samples = []
    for _ in range(num_bootstrap_samples):
        resampled = {}
        for dataset, scores in normalized.items():
            if not scores:
                raise ValueError("Each dataset must contain at least one score")
            resampled[dataset] = [rng.choice(scores) for _ in range(len(scores))]
        samples.append(macro_average(resampled))
    return {
        "point_estimate": point_estimate,
        "ci_low": _percentile(samples, 0.025),
        "ci_high": _percentile(samples, 0.975),
    }


def bootstrap_macro_delta(
    baseline_scores_by_dataset: Dict[str, Iterable[float]],
    comparison_scores_by_dataset: Dict[str, Iterable[float]],
    num_bootstrap_samples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    baseline = {dataset: list(scores) for dataset, scores in baseline_scores_by_dataset.items()}
    comparison = {dataset: list(scores) for dataset, scores in comparison_scores_by_dataset.items()}
    if baseline.keys() != comparison.keys():
        raise ValueError("Baseline and comparison datasets must match")

    point_estimate = macro_average(comparison) - macro_average(baseline)
    rng = random.Random(seed)
    samples = []
    for _ in range(num_bootstrap_samples):
        baseline_resampled = {}
        comparison_resampled = {}
        for dataset in baseline:
            baseline_scores = baseline[dataset]
            comparison_scores = comparison[dataset]
            if len(baseline_scores) != len(comparison_scores):
                raise ValueError("Paired bootstrap requires equal-length dataset score lists")
            indices = [rng.randrange(len(baseline_scores)) for _ in range(len(baseline_scores))]
            baseline_resampled[dataset] = [baseline_scores[idx] for idx in indices]
            comparison_resampled[dataset] = [comparison_scores[idx] for idx in indices]
        samples.append(macro_average(comparison_resampled) - macro_average(baseline_resampled))
    return {
        "point_estimate": point_estimate,
        "ci_low": _percentile(samples, 0.025),
        "ci_high": _percentile(samples, 0.975),
    }
