"""Train and evaluate SetFit on PAO metadata using shared 5-fold StratifiedGroupKFold.

Run from repo root:
  # Few-shot thesis ablation (16 unique examples/class, num_iterations=20)
  .venv\\Scripts\\python.exe scripts/classification/train_setfit.py --run-name setfit_v1_fewshot16 --samples-per-class 16 --batch-size 16 --num-iterations 20

  # Full-data 5-fold SetFit benchmark
  .venv\\Scripts\\python.exe scripts/classification/train_setfit.py --run-name setfit_v1_iter2 --batch-size 32 --num-iterations 2
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from intelligent_document_finetuning.classification.setfit_model import (
    DEFAULT_BASE_MODEL,
    SetFitClassifier,
)
from intelligent_document_finetuning.data.pao_metadata import load_pao_metadata
from intelligent_document_finetuning.evaluation.cv import evaluate_oof, make_cv_splits

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models" / "classification"


def subsample_per_class(train_df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Optional few-shot sampler: pick up to k unique-text examples per class."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in train_df.groupby("label", sort=True):
        # Prefer diverse texts first so few-shot doesn't pick k identical copies of "SICK LEAVE"
        unique_group = group.drop_duplicates(subset=["group_id"])
        source = unique_group if len(unique_group) >= k else group
        n_take = min(k, len(source))
        idx = rng.choice(source.index.to_numpy(), size=n_take, replace=False)
        parts.append(source.loc[idx])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-epochs", type=int, default=1)
    ap.add_argument("--num-iterations", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--samples-per-class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-final-fit", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    df = load_pao_metadata()
    splits = make_cv_splits(df, n_splits=5, seed=args.seed)

    oof_preds = np.empty(len(df), dtype=object)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        t_fold = time.time()
        train_df = df.iloc[tr_idx]
        if args.samples_per_class is not None:
            train_df = subsample_per_class(train_df, args.samples_per_class, seed=args.seed + fold)

        val_df = df.iloc[va_idx]
        print(f"\n=== Fold {fold}/5 | train={len(train_df)} | val={len(val_df)} ===", flush=True)

        clf = SetFitClassifier(
            base_model=args.base_model,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            num_iterations=args.num_iterations,
            max_steps=args.max_steps,
            seed=args.seed,
            use_amp=torch.cuda.is_available(),
        )
        clf.fit(train_df["document_text"], train_df["label"])

        preds = clf.predict(val_df["document_text"])
        oof_preds[va_idx] = preds

        fm = evaluate_oof(val_df["label"], preds)
        fold_summary = {
            "fold": fold,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "accuracy": round(float(fm["accuracy_all"]), 4),
            "macro_f1": round(float(fm["macro_f1_all"]), 4),
            "per_class_f1": fm["per_class_f1"],
            "seconds": round(time.time() - t_fold, 1),
        }
        fold_metrics.append(fold_summary)
        print(
            f"Fold {fold} done in {fold_summary['seconds']}s | "
            f"acc={fold_summary['accuracy']} | macro_f1={fold_summary['macro_f1']} | "
            f"per_class={fold_summary['per_class_f1']}",
            flush=True,
        )

        # Free GPU VRAM before starting the next fold
        del clf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    res = evaluate_oof(df["label"], oof_preds, has_conflict=df["has_label_conflict"])
    mean_acc = round(float(np.mean([f["accuracy"] for f in fold_metrics])), 4)
    std_acc = round(float(np.std([f["accuracy"] for f in fold_metrics])), 4)
    mean_f1 = round(float(np.mean([f["macro_f1"] for f in fold_metrics])), 4)
    std_f1 = round(float(np.std([f["macro_f1"] for f in fold_metrics])), 4)

    out_dir = MODELS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df.assign(predicted=oof_preds)[
        ["row_id", "document_text", "label", "predicted", "has_label_conflict"]
    ].to_parquet(out_dir / "oof_predictions.parquet", index=False)

    if not args.skip_final_fit:
        print("\n=== Fitting final SetFit model for saving ===", flush=True)
        final_train_df = (
            subsample_per_class(df, args.samples_per_class, seed=args.seed)
            if args.samples_per_class is not None
            else df
        )
        final_clf = SetFitClassifier(
            base_model=args.base_model,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            num_iterations=args.num_iterations,
            max_steps=args.max_steps,
            seed=args.seed,
            use_amp=torch.cuda.is_available(),
        ).fit(final_train_df["document_text"], final_train_df["label"])
        final_clf.save(out_dir / "model")

    cm = res["confusion_matrix"]
    run_meta = {
        "run_name": args.run_name,
        "params": vars(args),
        "n_rows": int(len(df)),
        "total_seconds": round(time.time() - t_start, 1),
        "cv_summary": {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
        },
        "oof_metrics": {
            "accuracy_all": round(float(res["accuracy_all"]), 4),
            "macro_f1_all": round(float(res["macro_f1_all"]), 4),
            "accuracy_no_conflict": round(float(res["accuracy_no_conflict"]), 4),
            "macro_f1_no_conflict": round(float(res["macro_f1_no_conflict"]), 4),
            "per_class_f1": res["per_class_f1"],
        },
        "confusion_matrix": {
            row: {col: int(cm.loc[row, col]) for col in cm.columns}
            for row in cm.index
        },
        "folds": fold_metrics,
    }
    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("\n================ FINAL SETFIT SUMMARY ================")
    print(f"Run: {args.run_name} | Total time: {run_meta['total_seconds']}s")
    print(f"Mean Accuracy: {mean_acc} (±{std_acc}) | Mean Macro-F1: {mean_f1} (±{std_f1})")
    print(
        f"OOF Macro-F1 (all): {run_meta['oof_metrics']['macro_f1_all']} | "
        f"(no conflict): {run_meta['oof_metrics']['macro_f1_no_conflict']}"
    )
    print("Per-class F1:", res["per_class_f1"])
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
