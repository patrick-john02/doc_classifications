"""Confidence threshold calibration for the PAO archiving agent's LLM fallback.

Why this file exists:
- In Intelligent-Document-Search, the archiving agent (Alon) auto-archives a document
  when classifier confidence >= threshold, and hands off to the LLM fallback otherwise.
- This module turns out-of-fold probabilities into a concrete trade-off table so your
  chosen threshold is backed by real precision/coverage numbers for your thesis.
"""

from typing import Sequence
import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def evaluate_thresholds(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    confidences: pd.Series | np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Compute auto-archive coverage, accuracy, and error-catch rate per threshold.

    Args:
        y_true: Ground-truth labels (length N).
        y_pred: Classifier's top predicted labels (length N).
        confidences: Top predicted probability max(probs) per sample (length N, in [0, 1]).
        thresholds: Confidence cutoffs to evaluate.

    Returns:
        DataFrame with one row per threshold and columns:
            threshold, auto_count, coverage_pct, auto_accuracy,
            fallback_count, fallback_pct, errors_caught_pct
    """
    true_arr = np.asarray(y_true)
    pred_arr = np.asarray(y_pred)
    conf_arr = np.asarray(confidences, dtype=float)

    correct = true_arr == pred_arr
    total_errors = int((~correct).sum())
    n_total = len(true_arr)

    rows: list[dict] = []

    for t in thresholds:
        auto_mask = conf_arr >= t
        fallback_mask = ~auto_mask

        auto_count = int(auto_mask.sum())
        fallback_count = int(fallback_mask.sum())
        auto_acc = float(correct[auto_mask].mean()) if auto_count > 0 else float("nan")

        caught_errors = int((fallback_mask & (~correct)).sum())
        errors_caught_pct = (
            (caught_errors / total_errors * 100.0) if total_errors > 0 else 100.0
        )

        rows.append({
            "threshold": round(float(t), 2),
            "auto_count": auto_count,
            "coverage_pct": round(auto_count / n_total * 100.0, 2),
            "auto_accuracy": round(auto_acc, 4) if not np.isnan(auto_acc) else None,
            "fallback_count": fallback_count,
            "fallback_pct": round(fallback_count / n_total * 100.0, 2),
            "errors_caught_pct": round(errors_caught_pct, 2),
        })

    return pd.DataFrame(rows)


def pick_recommended_threshold(
    table: pd.DataFrame,
    min_auto_accuracy: float = 0.97,
) -> dict | None:
    """Pick the lowest threshold (highest auto-archive coverage) meeting min_auto_accuracy."""
    valid = table.dropna(subset=["auto_accuracy"])
    passing = valid[valid["auto_accuracy"] >= min_auto_accuracy]
    if passing.empty:
        return None
    best = passing.sort_values("coverage_pct", ascending=False).iloc[0]
    return best.to_dict()
