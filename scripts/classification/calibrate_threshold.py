"""Calibrate the confidence threshold for auto-archiving vs LLM fallback.

The Calibration Math Toolbox

**Purpose:** Pure math functions (`evaluate_thresholds` and `pick_recommended_threshold`) with zero file I/O or CLI code. 
Given `(y_true, y_pred, confidences)`, it tests cutoffs `0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95` and computes:

- **`coverage_pct`:** % of documents auto-archived immediately (fast, $0 LLM cost).
- **`auto_accuracy`:** % of auto-archived documents that are actually right.
- **`fallback_pct`:** % of documents handed off to the LLM fallback.
- **`errors_caught_pct`:** % of classifier mistakes safely intercepted and sent to the LLM instead of misfiled!

**Why it is important:**
- The archiving agent (**Alon** in `Intelligent-Document-Search`) is a **hybrid ML + LLM system**: fast classifier handles clear cases, LLM handles uncertain ones.
- Set `T` too **low** (`0.50`) → cheap & fast, but misfiles ambiguous docs.
- Set `T` too **high** (`0.95`) → super accurate, but sends too many easy docs to the slower/costlier LLM.
- Keeping this math in `src/` lets TF-IDF, SetFit, and future OCR tests all use the exact same calibration logic.


Examples (run from repo root):
  # 1) Fast local calibration on TF-IDF + LogReg (~5 seconds)
  .venv\\Scripts\\python.exe scripts/classification/calibrate_threshold.py --model tfidf

  # 2) Calibrate from any saved SetFit run that has a 'confidence' column
  .venv\\Scripts\\python.exe scripts/classification/calibrate_threshold.py --oof-parquet models/classification/setfit_v2_e5_base/oof_predictions.parquet
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from intelligent_document_finetuning.classification.tfidf_logreg import TfidfLogRegClassifier
from intelligent_document_finetuning.data.pao_metadata import load_pao_metadata
from intelligent_document_finetuning.evaluation.cv import make_cv_splits
from intelligent_document_finetuning.evaluation.thresholds import (
    evaluate_thresholds,
    pick_recommended_threshold,
)


def run_tfidf_oof() -> pd.DataFrame:
    df = load_pao_metadata()
    splits = make_cv_splits(df, n_splits=5, seed=42)
    oof_preds = np.empty(len(df), dtype=object)
    oof_conf = np.zeros(len(df), dtype=float)

    for tr, va in splits:
        clf = TfidfLogRegClassifier(use_char_ngrams=True).fit(
            df["document_text"].iloc[tr], df["label"].iloc[tr]
        )
        probs = clf.predict_proba(df["document_text"].iloc[va])
        classes = list(clf.classes_)
        oof_preds[va] = [classes[i] for i in np.argmax(probs, axis=1)]
        oof_conf[va] = np.max(probs, axis=1)

    return df.assign(predicted=oof_preds, confidence=oof_conf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tfidf"], default="tfidf")
    ap.add_argument("--oof-parquet", type=Path, default=None)
    ap.add_argument("--target-accuracy", type=float, default=0.97)
    args = ap.parse_args()

    if args.oof_parquet is not None:
        oof = pd.read_parquet(args.oof_parquet)
        if "confidence" not in oof.columns:
            raise ValueError(
                f"{args.oof_parquet} has no 'confidence' column. "
                "Re-run train_setfit.py (updated to save OOF confidence)."
            )
        source_name = str(args.oof_parquet)
    else:
        oof = run_tfidf_oof()
        source_name = "TfidfLogRegClassifier (word + char_wb, 5-fold OOF)"

    print(f"\n=== Threshold Calibration: {source_name} (n={len(oof)}) ===")
    table_all = evaluate_thresholds(oof["label"], oof["predicted"], oof["confidence"])
    print("\nAll rows:")
    print(table_all.to_string(index=False))

    rec_all = pick_recommended_threshold(table_all, min_auto_accuracy=args.target_accuracy)
    print(f"\nRecommended threshold for >= {args.target_accuracy:.0%} auto-accuracy (all rows):")
    print(" ", rec_all if rec_all else "No threshold met target accuracy.")

    if "has_label_conflict" in oof.columns:
        clean = oof[~oof["has_label_conflict"]]
        table_clean = evaluate_thresholds(clean["label"], clean["predicted"], clean["confidence"])
        print(f"\nExcluding identical-text conflict rows (n={len(clean)}):")
        print(table_clean.to_string(index=False))

        rec_clean = pick_recommended_threshold(table_clean, min_auto_accuracy=args.target_accuracy)
        print(f"\nRecommended threshold for >= {args.target_accuracy:.0%} auto-accuracy (no conflicts):")
        print(" ", rec_clean if rec_clean else "No threshold met target accuracy.")


if __name__ == "__main__":
    main()
