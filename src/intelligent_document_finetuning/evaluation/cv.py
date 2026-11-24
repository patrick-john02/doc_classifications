"""Shared cross-validation and evaluation metrics for all PAO classifiers.

Why this file exists:
- Every model (TF-IDF + LogReg, LinearSVC, SetFit) MUST be evaluated on the exact
  same 5 folds and the exact same metrics so comparisons are 100% fair.
- Learn Mode exercise: fill in the 2 short TODO blocks below (~15 lines total).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold


def make_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build leakage-free StratifiedGroupKFold splits from df['group_id'].

    Returns:
        List of (train_idx, val_idx) numpy index arrays, one tuple per fold.
    """
    # TODO 1 (Learn Mode — ~7 lines):
    # 1. Create a StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # 2. Loop over cv.split(df["document_text"], df["label"], groups=df["group_id"])
    # 3. For each fold, assert that train and val share ZERO group_ids (leakage check!)
    #    Hint: set(df["group_id"].iloc[train_idx]).intersection(df["group_id"].iloc[val_idx])
    # 4. Collect (train_idx, val_idx) into a list called `splits` and return it.

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits:list[tuple[np.ndarray, np.ndarray]] = []
    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(
            df["document_text"].to_numpy(),
            df["label"],
            groups=df["group_id"],
        ),
        start=1
        
    ):
        
        print(f"Fold {fold}: Train={len(train_idx)}, Validation={len(val_idx)}")

        train_groups = set(df["group_id"].iloc[train_idx])
        val_groups = set(df["group_id"].iloc[val_idx])


        overlap = train_groups.intersection(val_groups)
        assert len(overlap) == 0, f"Group leakage detected in fold {fold}!"
        splits.append((train_idx, val_idx))

    return splits


def evaluate_oof(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    has_conflict: pd.Series | np.ndarray | None = None,
) -> dict:
    """Compute standard out-of-fold metrics shared across all classifiers.

    Returns a dictionary containing:
        - accuracy_all, macro_f1_all
        - accuracy_no_conflict, macro_f1_no_conflict (if has_conflict is provided)
        - per_class_f1: dict mapping each class name -> rounded F1 score
        - confusion_matrix: DataFrame (rows = actual_<class>, cols = pred_<class>)
    """
    
    labels = sorted(pd.Series(y_true).unique())

    # TODO 2 (Learn Mode — ~8 lines):
    # 1. Compute `acc_all = accuracy_score(y_true, y_pred)`
    #    and `macro_all = f1_score(y_true, y_pred, average="macro")`
    # 2. Compute `report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)`
    #    and extract `per_class_f1 = {lbl: round(float(report[lbl]["f1-score"]), 4) for lbl in labels}`
    # 3. Build a DataFrame `cm` from `confusion_matrix(y_true, y_pred, labels=labels)`
    #    with index=[f"actual_{l}" for l in labels] and columns=[f"pred_{l}" for l in labels]
    # 4. Put them into `metrics = {"accuracy_all": ..., "macro_f1_all": ..., "per_class_f1": ..., "confusion_matrix": cm}`
    # 5. If `has_conflict is not None`, compute accuracy & macro-F1 on `mask = ~np.asarray(has_conflict, dtype=bool)`
    #    and add "accuracy_no_conflict" and "macro_f1_no_conflict" to `metrics`, then return `metrics`.

    acc_all=accuracy_score(y_true, y_pred)
    macro_all=f1_score(y_true, y_pred, average="macro")

    report=classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    per_class_f1={}

    for lbl in labels:
        class_metrics=report[lbl]

        if not isinstance(class_metrics, dict):
            raise TypeError(f"Expected metrics dictionary for {lbl}")
        per_class_f1[lbl]=round(float(class_metrics["f1-score"]),4)

    cm=pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"actual_{l}" for l in labels] ,
        columns=[f"pred_{l}" for l in labels]

    )

    metrics = {
        "accuracy_all": acc_all,
        "macro_f1_all":macro_all,
        "per_class_f1": per_class_f1,
        "confusion_matrix":cm
    }



    if has_conflict is not None:
        mask = ~np.asarray(has_conflict, dtype=bool)

        if mask.any():
            true_clean=np.asarray(y_true)[mask]
            pred_clean=np.asarray(y_pred)[mask]

            metrics["accuracy_no_conflict"] = accuracy_score(true_clean, pred_clean)
            metrics["macro_f1_no_conflict"] = f1_score(
                true_clean, pred_clean, average="macro", zero_division=0
            )
        
    return metrics


