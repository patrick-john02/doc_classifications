"""Human-in-the-Loop (HITL) feedback merging, OCR artifact cleaning, and promotion gating.

Why this file exists:
- Intelligent-Document-Search logs every human-confirmed/corrected document category
  into a feedback table.
- This module merges exported feedback with the existing real-OCR corpus safely:
    1. Cleans RapidOCR '[?...]' and '|' formatting artifacts consistently.
    2. Deduplicates by doc_id (human correction always overrides older labels).
    3. Enforces a minimum sample count per class (default >= 8) so 1-2 stray edits
       never destabilize 5-fold CV.
    4. Runs a Champion-vs-Challenger promotion gate before replacing a live model.
"""

from pathlib import Path
import re
import pandas as pd


def clean_rapidocr_artifacts(text: str) -> str:
    """Strip RapidOCR '[?...]' low-confidence markers and '|' table column separators."""
    text = re.sub(r"\[\?|\]|\|", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def load_feedback_file(path: Path | str) -> pd.DataFrame:
    """Load an exported HITL feedback CSV or Parquet file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feedback file not found: {p}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported feedback file format '{p.suffix}'. Use .csv or .parquet.")


def _normalize_columns(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Map common feedback/corpus column names onto canonical (doc_id, filename, label, text)."""
    out = df.copy()

    # Resolve text column
    if text_col in out.columns:
        out["text"] = out[text_col]
    elif "ocr_text_page1" in out.columns:
        out["text"] = out["ocr_text_page1"]
    elif "ocr_text" in out.columns:
        out["text"] = out["ocr_text"]
    elif "text" not in out.columns:
        raise ValueError(
            f"Could not find a text column (expected '{text_col}', 'ocr_text_page1', 'ocr_text', or 'text')."
        )

    # Resolve label column (prefer human_label from HITL feedback if present)
    if "human_label" in out.columns:
        out["label"] = out["human_label"]
    elif "label" not in out.columns:
        raise ValueError("Could not find a label column ('human_label' or 'label').")

    if "doc_id" not in out.columns:
        raise ValueError("Dataset must contain a 'doc_id' column for deduplication.")

    if "filename" not in out.columns:
        out["filename"] = out["doc_id"].astype(str)

    out["doc_id"] = out["doc_id"].astype(str).str.strip()
    out["label"] = out["label"].fillna("").astype(str).str.strip()
    out["clean_text"] = out["text"].fillna("").astype(str).map(clean_rapidocr_artifacts).str.upper()

    valid = (out["doc_id"] != "") & (out["label"] != "") & (out["clean_text"] != "")
    return out.loc[valid, ["doc_id", "filename", "label", "clean_text"]].reset_index(drop=True)


def merge_feedback_dataset(
    base_df: pd.DataFrame,
    feedback_df: pd.DataFrame | None = None,
    text_col: str = "text_page1",
    min_class_samples: int = 8,
) -> tuple[pd.DataFrame, dict]:
    """Merge base real-OCR scans with optional HITL feedback and enforce min class size.

    Deduplication rule: keep='last' so newer human corrections override older rows
    with the same doc_id.
    """
    base_norm = _normalize_columns(base_df, text_col=text_col)
    base_norm["source"] = "base_ocr"

    if feedback_df is not None and len(feedback_df) > 0:
        fb_norm = _normalize_columns(feedback_df, text_col=text_col)
        fb_norm["source"] = "hitl_feedback"
        combined = pd.concat([base_norm, fb_norm], ignore_index=True)
        n_feedback = len(fb_norm)
    else:
        combined = base_norm
        n_feedback = 0

    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["doc_id"], keep="last").reset_index(drop=True)
    n_overridden = before_dedup - len(combined)

    counts = combined["label"].value_counts()
    skipped_classes = {
        str(lbl): int(cnt) for lbl, cnt in counts.items() if cnt < min_class_samples
    }
    eligible_labels = set(counts[counts >= min_class_samples].index)
    filtered = combined[combined["label"].isin(eligible_labels)].reset_index(drop=True)

    stats = {
        "base_rows": int(len(base_norm)),
        "feedback_rows": int(n_feedback),
        "overridden_doc_ids": int(n_overridden),
        "final_rows": int(len(filtered)),
        "min_class_samples": int(min_class_samples),
        "included_classes": {str(k): int(v) for k, v in filtered["label"].value_counts().items()},
        "skipped_classes": skipped_classes,
    }
    return filtered, stats


def decide_promotion(
    challenger_best: dict,
    champion_report: dict | None,
    min_f1_delta: float = -0.005,
    min_new_class_f1: float = 0.75,
) -> dict:
    """Champion-vs-Challenger gate: decide whether to PROMOTE or HOLD a retrained model.

    Args:
        challenger_best: Best model summary dict from the new run (macro_f1, per_class_f1).
        champion_report: Parsed run.json from the currently deployed champion model, or None.
        min_f1_delta: Allowed Macro-F1 tolerance when class sets are identical (default -0.005).
        min_new_class_f1: Minimum F1 required on every newly unlocked class (e.g., AOsMEMOs).
    """
    challenger_f1 = float(challenger_best["macro_f1"])
    challenger_classes = set(challenger_best["per_class_f1"].keys())

    if champion_report is None:
        return {
            "decision": "PROMOTE",
            "reason": "no_existing_champion",
            "challenger_macro_f1": round(challenger_f1, 4),
            "champion_macro_f1": None,
            "new_classes": sorted(challenger_classes),
        }

    champ_best_name = champion_report.get("production_candidate") or champion_report.get("best_model")
    champ_results = champion_report.get("results", [])
    champ_entry = next((r for r in champ_results if r["model"] == champ_best_name), champ_results[0])
    champion_f1 = float(champ_entry["macro_f1"])
    champion_classes = set(champ_entry["per_class_f1"].keys())

    new_classes = sorted(challenger_classes - champion_classes)
    dropped_classes = sorted(champion_classes - challenger_classes)

    if dropped_classes:
        return {
            "decision": "HOLD",
            "reason": f"missing_champion_classes:{dropped_classes}",
            "challenger_macro_f1": round(challenger_f1, 4),
            "champion_macro_f1": round(champion_f1, 4),
            "new_classes": new_classes,
        }

    if new_classes:
        weak_new = {
            c: float(challenger_best["per_class_f1"][c])
            for c in new_classes
            if float(challenger_best["per_class_f1"][c]) < min_new_class_f1
        }
        if weak_new:
            return {
                "decision": "HOLD",
                "reason": f"new_class_f1_below_{min_new_class_f1}:{weak_new}",
                "challenger_macro_f1": round(challenger_f1, 4),
                "champion_macro_f1": round(champion_f1, 4),
                "new_classes": new_classes,
            }
        if challenger_f1 < champion_f1 - 0.05:
            return {
                "decision": "HOLD",
                "reason": "overall_macro_f1_regressed_with_new_class",
                "challenger_macro_f1": round(challenger_f1, 4),
                "champion_macro_f1": round(champion_f1, 4),
                "new_classes": new_classes,
            }
        return {
            "decision": "PROMOTE",
            "reason": f"unlocked_new_classes:{new_classes}",
            "challenger_macro_f1": round(challenger_f1, 4),
            "champion_macro_f1": round(champion_f1, 4),
            "new_classes": new_classes,
        }

    delta = round(challenger_f1 - champion_f1, 4)
    if delta >= min_f1_delta:
        return {
            "decision": "PROMOTE",
            "reason": f"macro_f1_delta_{delta:+.4f}",
            "challenger_macro_f1": round(challenger_f1, 4),
            "champion_macro_f1": round(champion_f1, 4),
            "new_classes": [],
        }

    return {
        "decision": "HOLD",
        "reason": f"macro_f1_regressed_{delta:+.4f}",
        "challenger_macro_f1": round(challenger_f1, 4),
        "champion_macro_f1": round(champion_f1, 4),
        "new_classes": [],
    }
