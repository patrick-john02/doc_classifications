"""Path B: Few-shot Real-OCR training, HITL feedback merging, and Champion-vs-Challenger gating.

Why this script exists:
- Phase 1 (01_extract_and_eval_real_ocr.py) proved that models trained ONLY on short
  Excel logbook metadata collapse on real scanned PDFs (44.6% accuracy for metadata SetFit)
  because real forms are 75-170 words long with RapidOCR spacing quirks (e.g., APPLICATIONFORLEAVE).
- This script benchmarks 3 models trained directly on your real RapidOCR scans (+ optional
  Human-in-the-Loop corrections exported from Intelligent-Document-Search) using 5-fold
  StratifiedKFold:
    1. tfidf_char_real_ocr          -> TF-IDF (word + char_wb) + LogReg baseline
    2. setfit_fresh_real_ocr        -> Few-shot SetFit starting from pretrained MiniLM
    3. setfit_adapted_from_metadata -> Domain-adapted SetFit starting from setfit_v1_iter2
- Then it calibrates the LLM-fallback confidence threshold, runs a Champion-vs-Challenger
  promotion gate, and saves versioned artifacts to models/classification/<run_name>/.

Run from repo root:
    # Standard real-OCR training
    .venv\\Scripts\\python.exe scripts/pao_ocr/02_train_real_ocr.py --run-name real_ocr_v1

    # Retraining with HITL feedback exported from Intelligent-Document-Search + safety gate
    .venv\\Scripts\\python.exe scripts/pao_ocr/02_train_real_ocr.py ^
        --run-name real_ocr_v2_hitl ^
        --feedback-file data/interim/hitl_feedback.csv ^
        --min-class-samples 8 ^
        --champion-run models/classification/real_ocr_v1/run.json
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from intelligent_document_finetuning.classification.setfit_model import (
    DEFAULT_BASE_MODEL,
    SetFitClassifier,
)
from intelligent_document_finetuning.classification.tfidf_logreg import TfidfLogRegClassifier
from intelligent_document_finetuning.data.feedback import (
    decide_promotion,
    load_feedback_file,
    merge_feedback_dataset,
)
from intelligent_document_finetuning.evaluation.cv import evaluate_oof
from intelligent_document_finetuning.evaluation.thresholds import (
    evaluate_thresholds,
    pick_recommended_threshold,
)

# Paths anchored to repo root so the script works regardless of working directory
ROOT = Path(__file__).resolve().parents[2]
OCR_PARQUET = ROOT / "data" / "processed" / "pao_real_ocr_v1.parquet"
METADATA_SETFIT = ROOT / "models" / "classification" / "setfit_v1_iter2" / "model"
MODELS_DIR = ROOT / "models" / "classification"


def run_cv_model(
    name: str,
    df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    make_model,
    text_col: str,
) -> tuple[dict, pd.DataFrame]:
    """Run 5-fold cross-validation for one model and collect honest OOF probabilities."""
    t0 = time.time()
    class_names = sorted(df["label"].unique())
    oof_preds = np.empty(len(df), dtype=object)
    oof_conf = np.zeros(len(df), dtype=float)
    oof_probs = np.zeros((len(df), len(class_names)), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        # Fresh model per fold so weights never leak across folds
        clf = make_model()
        clf.fit(df[text_col].iloc[tr_idx], df["label"].iloc[tr_idx])

        # Single predict_proba() pass gives both top label and calibrated confidence
        probs = clf.predict_proba(df[text_col].iloc[va_idx])
        fold_classes = list(clf.classes_)
        oof_preds[va_idx] = [fold_classes[i] for i in np.argmax(probs, axis=1)]
        oof_conf[va_idx] = np.max(probs, axis=1)
        for col_i, cname in enumerate(fold_classes):
            oof_probs[va_idx, class_names.index(cname)] = probs[:, col_i]

        # Free GPU VRAM between folds so 5 folds + final fit never OOM on an 8 GB GPU
        del clf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    res = evaluate_oof(df["label"], oof_preds)
    cm = res["confusion_matrix"]
    summary = {
        "model": name,
        "accuracy": round(float(res["accuracy_all"]), 4),
        "macro_f1": round(float(res["macro_f1_all"]), 4),
        "per_class_f1": res["per_class_f1"],
        "seconds": round(time.time() - t0, 1),
        "confusion_matrix": {
            row: {col: int(cm.loc[row, col]) for col in cm.columns} for row in cm.index
        },
    }

    oof_df = df[["doc_id", "filename", "label"]].copy()
    oof_df["predicted"] = oof_preds
    oof_df["confidence"] = np.round(oof_conf, 4)
    for col_i, cname in enumerate(class_names):
        oof_df[f"prob_{cname}"] = np.round(oof_probs[:, col_i], 4)

    print(f"\n=== {name} ({summary['seconds']}s) ===", flush=True)
    print(
        f"Accuracy: {summary['accuracy']} | Macro-F1: {summary['macro_f1']} | "
        f"Per-class: {summary['per_class_f1']}",
        flush=True,
    )
    print(cm, flush=True)
    return summary, oof_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="real_ocr_v1")
    ap.add_argument("--ocr-parquet", type=Path, default=OCR_PARQUET)
    ap.add_argument(
        "--feedback-file",
        type=Path,
        default=None,
        help="Optional .csv or .parquet exported from Intelligent-Document-Search's HITL feedback table.",
    )
    ap.add_argument(
        "--min-class-samples",
        type=int,
        default=8,
        help="Minimum verified scans required before a class is included in training.",
    )
    ap.add_argument(
        "--champion-run",
        type=Path,
        default=None,
        help="Optional path to current live champion run.json for Champion-vs-Challenger promotion gating.",
    )
    ap.add_argument(
        "--text-col",
        default="text_page1",
        choices=["text_full", "text_page1", "text_header50"],
        help="Page 1 contains the form title/header without noisy trailing attachments.",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-iterations", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1) Load base RapidOCR scans + optional HITL feedback, deduplicating by doc_id
    base_df = pd.read_parquet(args.ocr_parquet)
    feedback_df = load_feedback_file(args.feedback_file) if args.feedback_file else None
    df, merge_stats = merge_feedback_dataset(
        base_df,
        feedback_df=feedback_df,
        text_col=args.text_col,
        min_class_samples=args.min_class_samples,
    )
    if df["label"].nunique() < 2:
        raise ValueError(f"Need at least 2 classes with >= {args.min_class_samples} samples. Stats: {merge_stats}")

    print(f"Dataset ready: {merge_stats['final_rows']} scans | classes: {merge_stats['included_classes']}")
    if merge_stats["skipped_classes"]:
        print(f"[guard] Skipped under-represented classes (< {args.min_class_samples}): {merge_stats['skipped_classes']}")

    # 2) Build shared 5-fold StratifiedKFold splits so all candidates face identical test folds
    n_splits = min(5, int(df["label"].value_counts().min()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    splits = list(skf.split(df["clean_text"], df["label"]))

    # 3) Define candidates: TF-IDF char baseline vs Fresh SetFit vs Metadata-Adapted SetFit
    candidates: list[tuple[str, object]] = [
        ("tfidf_char_real_ocr", lambda: TfidfLogRegClassifier(use_char_ngrams=True, min_df=1)),
        (
            "setfit_fresh_real_ocr",
            lambda: SetFitClassifier(
                base_model=DEFAULT_BASE_MODEL,
                batch_size=args.batch_size,
                num_epochs=1,
                num_iterations=args.num_iterations,
                seed=args.seed,
                use_amp=torch.cuda.is_available(),
            ),
        ),
    ]
    if METADATA_SETFIT.exists():
        candidates.append((
            "setfit_adapted_from_metadata",
            lambda: SetFitClassifier(
                base_model=str(METADATA_SETFIT),
                batch_size=args.batch_size,
                num_epochs=1,
                num_iterations=args.num_iterations,
                seed=args.seed,
                use_amp=torch.cuda.is_available(),
            ),
        ))

    # 4) Run identical CV for every candidate
    summaries = []
    oof_map: dict[str, pd.DataFrame] = {}
    factory_map = dict(candidates)

    for name, factory in candidates:
        s, oof_df = run_cv_model(name, df, splits, factory, text_col="clean_text")
        summaries.append(s)
        oof_map[name] = oof_df

    # 5) Select the winning model by Macro-F1 and calibrate its LLM-fallback threshold
    best = max(summaries, key=lambda x: x["macro_f1"])
    best_name = best["model"]
    best_oof = oof_map[best_name]

    thresh_table = evaluate_thresholds(
        best_oof["label"], best_oof["predicted"], best_oof["confidence"]
    )
    rec = pick_recommended_threshold(thresh_table, min_auto_accuracy=0.95)
    print(f"\n=== Threshold Calibration on Real OCR ({best_name}) ===")
    print(thresh_table.to_string(index=False))
    print("Recommended threshold (>=95% auto-accuracy):", rec)

    # 6) Champion-vs-Challenger promotion gate
    champion_report = (
        json.loads(args.champion_run.read_text(encoding="utf-8"))
        if args.champion_run and args.champion_run.exists()
        else None
    )
    promotion = decide_promotion(best, champion_report)
    print(f"\n=== Promotion Gate: {promotion['decision']} ({promotion['reason']}) ===")

    # 7) Fit final winning model on ALL merged scans and save reproducible artifacts
    out_dir = MODELS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_oof.to_parquet(out_dir / "oof_predictions.parquet", index=False)

    print(f"\n=== Fitting final winning model ({best_name}) on all {len(df)} scans ===", flush=True)
    final_clf = factory_map[best_name]().fit(df["clean_text"], df["label"])
    if isinstance(final_clf, SetFitClassifier):
        final_clf.save(out_dir / "model")
    else:
        final_clf.save(out_dir / "model.joblib")

    report = {
        "run_name": args.run_name,
        "n_docs": int(len(df)),
        "class_counts": merge_stats["included_classes"],
        "merge_stats": merge_stats,
        "text_col": args.text_col,
        "n_splits": n_splits,
        "results": summaries,
        "best_model": best_name,
        "recommended_threshold_95": rec,
        "promotion": promotion,
    }
    (out_dir / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved real-OCR experiment & final model to: {out_dir}")


if __name__ == "__main__":
    main()
