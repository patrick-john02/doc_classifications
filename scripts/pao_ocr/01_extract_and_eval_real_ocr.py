"""Extract real scanned PAO PDFs via RapidOCR and run Stage 1 zero-shot transfer evaluation.

Put your scanned PDFs into:
    data/raw/pao_scans/CLEARANCE/
    data/raw/pao_scans/COMMUNICATION/
    data/raw/pao_scans/LEAVES/
    data/raw/pao_scans/AOsMEMOs/      (optional — add 8-10 whenever available)

Run from repo root:
    .venv\\Scripts\\python.exe scripts/pao_ocr/01_extract_and_eval_real_ocr.py
    # Re-run evaluation without re-OCR'ing PDFs:
    .venv\\Scripts\\python.exe scripts/pao_ocr/01_extract_and_eval_real_ocr.py --reuse-cache
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from intelligent_document_finetuning.classification.setfit_model import SetFitClassifier
from intelligent_document_finetuning.evaluation.cv import evaluate_oof
from intelligent_document_finetuning.ocr.client import ocr_images

ROOT = Path(__file__).resolve().parents[2]
SCANS_DIR = ROOT / "data" / "raw" / "pao_scans"
OUT_PARQUET = ROOT / "data" / "processed" / "pao_real_ocr_v1.parquet"
OUT_JSON = ROOT / "models" / "classification" / "real_ocr_eval_v1.json"

SUPPORTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Normalize common folder name variations to the canonical 4 thesis labels
FOLDER_TO_LABEL = {
    "aosmemos": "AOsMEMOs",
    "ao_memo": "AOsMEMOs",
    "memos": "AOsMEMOs",
    "clearance": "CLEARANCE",
    "clearances": "CLEARANCE",
    "communication": "COMMUNICATION",
    "communications": "COMMUNICATION",
    "leave": "LEAVES",
    "leaves": "LEAVES",
}

PAGE_HEADER_RE = re.compile(r"^##\s*Page\s+(\d+)(?:\s*\[Scanned Page\])?\s*$", re.MULTILINE)


def discover_scan_files(scans_dir: Path) -> list[tuple[Path, str]]:
    """Find all supported scan files grouped by class subfolder."""
    items: list[tuple[Path, str]] = []
    if not scans_dir.exists():
        return items
    for sub in sorted(scans_dir.iterdir()):
        if not sub.is_dir():
            continue
        canonical = FOLDER_TO_LABEL.get(sub.name.strip().lower())
        if canonical is None:
            print(f"[skip] Unrecognized folder '{sub.name}' (expected CLEARANCE, COMMUNICATION, LEAVES, AOsMEMOs)")
            continue
        for f in sorted(sub.rglob("*")):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
                items.append((f, canonical))
    return items


def clean_ocr_text(text: str) -> str:
    """Remove '## Page N' markers and collapse whitespace."""
    text = PAGE_HEADER_RE.sub(" ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_page1_text(raw_text: str) -> str:
    """Keep only Page 1 content (where form titles & document headers appear)."""
    matches = list(PAGE_HEADER_RE.finditer(str(raw_text)))
    if not matches:
        return clean_ocr_text(raw_text)
    start = matches[0].end()
    end = matches[1].start() if len(matches) > 1 else len(raw_text)
    return clean_ocr_text(raw_text[start:end])


def first_n_words(text: str, n: int = 50) -> str:
    return " ".join(str(text).split()[:n])


def extract_or_load_ocr(scans_dir: Path, out_parquet: Path, reuse_cache: bool) -> pd.DataFrame:
    if reuse_cache and out_parquet.exists():
        df = pd.read_parquet(out_parquet)
        print(f"[cache] Loaded {len(df)} OCR'd real scans from {out_parquet}")
        return df

    items = discover_scan_files(scans_dir)
    if not items:
        raise FileNotFoundError(
            f"No scan files found under {scans_dir}.\n"
            "Drop your PDFs into:\n"
            "  data/raw/pao_scans/CLEARANCE/\n"
            "  data/raw/pao_scans/COMMUNICATION/\n"
            "  data/raw/pao_scans/LEAVES/\n"
            "  data/raw/pao_scans/AOsMEMOs/  (optional)"
        )

    print(f"[ocr] Running RapidOCR (via Intelligent-Document-Search) on {len(items)} files...")
    paths = [p for p, _ in items]
    labels = [lbl for _, lbl in items]
    ocr_rows = ocr_images(paths)

    records = []
    for (path, label), r in zip(items, ocr_rows):
        raw_text = r.get("text", "") or ""
        full_clean = clean_ocr_text(raw_text)
        page1_clean = extract_page1_text(raw_text)
        header_50 = first_n_words(page1_clean, 50)
        records.append({
            "doc_id": f"{label}-{path.stem}",
            "filename": path.name,
            "label": label,
            "ocr_raw": raw_text,
            "text_full": full_clean.upper(),
            "text_page1": page1_clean.upper(),
            "text_header50": header_50.upper(),
            "words_full": len(full_clean.split()),
            "words_page1": len(page1_clean.split()),
            "is_scanned": bool(r.get("is_scanned", True)),
            "ocr_score": round(float(r.get("ocr_score", 0.0)), 4),
            "extractor": str(r.get("extractor", "unknown")),
            "error": r.get("error"),
        })

    df = pd.DataFrame(records)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"[ocr] Saved {len(df)} extracted scans to {out_parquet}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans-dir", type=Path, default=SCANS_DIR)
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument(
        "--setfit-dir",
        type=Path,
        default=ROOT / "models" / "classification" / "setfit_v1_iter2" / "model",
    )
    ap.add_argument(
        "--tfidf-path",
        type=Path,
        default=ROOT / "models" / "classification" / "tfidf_logreg_v2" / "best_model.joblib",
    )
    args = ap.parse_args()

    df = extract_or_load_ocr(args.scans_dir, OUT_PARQUET, args.reuse_cache)

    print("\n=== Real Scanned PAO Corpus Summary ===")
    print(df["label"].value_counts().to_string())
    print(
        f"Mean OCR score: {df['ocr_score'].mean():.3f} | "
        f"Median words (full): {df['words_full'].median():.0f} | "
        f"Median words (page1): {df['words_page1'].median():.0f} | "
        f"Errors: {df['error'].notna().sum()}"
    )

    usable = df[df["text_full"].str.len() > 0].reset_index(drop=True)
    if len(usable) == 0:
        raise RuntimeError("All OCR outputs were empty. Check PDF extraction errors above.")

    # Load metadata-trained models for zero-shot transfer test
    models: dict[str, object] = {}
    if args.tfidf_path.exists():
        models["tfidf_v2_best"] = joblib.load(args.tfidf_path)
    if args.setfit_dir.exists():
        models["setfit_v1_iter2"] = SetFitClassifier.load(args.setfit_dir)

    views = ["text_full", "text_page1", "text_header50"]
    results_summary = []

    for mname, model in models.items():
        for view in views:
            texts = usable[view].tolist()
            preds = np.asarray(model.predict(texts), dtype=object)
            metrics = evaluate_oof(usable["label"], preds)
            row = {
                "model": mname,
                "view": view,
                "accuracy": round(float(metrics["accuracy_all"]), 4),
                "macro_f1": round(float(metrics["macro_f1_all"]), 4),
                "per_class_f1": metrics["per_class_f1"],
            }
            results_summary.append(row)
            print(f"\n--- {mname} | {view} ---")
            print(f"Accuracy: {row['accuracy']} | Macro-F1: {row['macro_f1']} | Per-class: {row['per_class_f1']}")
            print(metrics["confusion_matrix"])

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "n_docs": int(len(usable)),
                "class_counts": {k: int(v) for k, v in usable["label"].value_counts().items()},
                "results": results_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved zero-shot real-OCR evaluation report to: {OUT_JSON}")


if __name__ == "__main__":
    main()
