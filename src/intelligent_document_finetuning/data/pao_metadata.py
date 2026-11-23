"""Loader for the cleaned PAO logbook metadata dataset.

Why this file exists (instead of loading parquet inside every notebook):
- One single source of truth for the default dataset path and filtering rules.
- Works whether called from notebooks/, scripts/, or another repo.
- Guarantees TF-IDF, LinearSVC, and SetFit all read the exact same rows.
"""

from pathlib import Path
import pandas as pd

# Repo root is 3 levels above src/intelligent_document_finetuning/data/pao_metadata.py
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PARQUET = REPO_ROOT / "data" / "processed" / "pao_metadata_v2.parquet"

REQUIRED_COLUMNS = (
    "row_id",
    "document_text",
    "label",
    "review_status",
    "group_id",
    "has_label_conflict",
)


def load_pao_metadata(
    path: Path | str = DEFAULT_PARQUET,
    exclude_domain_review: bool = True,
    exclude_conflicts: bool = False,
) -> pd.DataFrame:
    """Load the processed PAO metadata parquet and apply standard filters.

    Args:
        path: Path to the processed .parquet file (defaults to v2).
        exclude_domain_review: Drop rows marked 'needs_domain_review' (recommended).
        exclude_conflicts: Optionally drop rows flagged with identical-text
            CLEARANCE vs LEAVES conflicts.

    Returns:
        A clean DataFrame with reset index and verified required columns.
    """
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {parquet_path}. "
            "Run notebooks/01_pao_metadata_preparation.ipynb first."
        )

    df = pd.read_parquet(parquet_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset {parquet_path} is missing required columns: {missing}")

    if exclude_domain_review:
        df = df[df["review_status"] != "needs_domain_review"]

    if exclude_conflicts:
        df = df[~df["has_label_conflict"]]

    return df.reset_index(drop=True)
