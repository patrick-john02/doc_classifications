"""Task 3: build the RVL-CDIP OCR corpus (v1).

sample -> extract from tar -> OCR (resumable) -> parquet
Run:  python scripts/07_prepare_document_corpus.py --per-class 100
"""
import argparse
import json
import random
import re
import sys
import tarfile
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ocr_client import ocr_images

RAW = ROOT / "data" / "raw" / "rvl_cdip"
TAR = RAW / "data" / "rvl-cdip.tar.gz"
LABELS = RAW / "data" / "train.txt"
IMG_DIR = RAW / "subset_v1"
INTERIM = ROOT / "data" / "interim"
MANIFEST = INTERIM / "rvl_cdip_v1_manifest.csv"
OCR_LOG = INTERIM / "rvl_cdip_v1_ocr.jsonl"
OUT = ROOT / "data" / "processed" / "rvl_cdip_v1.parquet"

# RVL-CDIP label ids 0-15
CLASSES = ["letter", "form", "email", "handwritten", "advertisement",
           "scientific_report", "scientific_publication", "specification",
           "file_folder", "news_article", "budget", "invoice",
           "presentation", "questionnaire", "resume", "memo"]

KEY_RE = re.compile(r"(images[a-z]/.+\.tif)$", re.I)


def build_manifest(per_class: int, seed: int) -> pd.DataFrame:
    if MANIFEST.exists():
        print(f"[1/4] reusing manifest {MANIFEST.name} (delete it to resample)")
        return pd.read_csv(MANIFEST, dtype={"doc_id": str})

    by_label = defaultdict(list)
    for line in LABELS.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            by_label[int(parts[1])].append(parts[0])

    rng = random.Random(seed)
    rows, seen = [], set()
    for label in sorted(by_label):
        paths = by_label[label][:]
        rng.shuffle(paths)
        picked = 0
        for p in paths:
            doc_id = Path(p).stem
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rows.append({"doc_id": doc_id, "tar_key": p,
                         "label_id": label, "label": CLASSES[label]})
            picked += 1
            if picked == per_class:
                break

    df = pd.DataFrame(rows)
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, index=False)
    print(f"[1/4] sampled {len(df)} docs ({per_class}/class, seed={seed})")
    return df


def extract_images(manifest: pd.DataFrame) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    want = {r.tar_key: r.doc_id for r in manifest.itertuples()
            if not (IMG_DIR / f"{r.doc_id}.tif").exists()}
    if not want:
        print("[2/4] all images already extracted")
        return

    print(f"[2/4] extracting {len(want)} images (one pass over the tar, be patient)...")
    with tarfile.open(TAR, "r|gz") as tar:          # streaming: no full unpack
        for member in tar:
            if not member.isfile():
                continue
            m = KEY_RE.search(member.name)
            if not m or m.group(1) not in want:
                continue
            doc_id = want.pop(m.group(1))
            (IMG_DIR / f"{doc_id}.tif").write_bytes(tar.extractfile(member).read())
            if len(want) % 100 == 0:
                print(f"   {len(want)} left")
            if not want:
                break
    if want:
        print(f"   WARNING: {len(want)} not found in tar, e.g. {next(iter(want))}")


def run_ocr(manifest: pd.DataFrame, batch_size: int) -> None:
    done = set()
    if OCR_LOG.exists():
        with OCR_LOG.open(encoding="utf-8") as f:
            done = {json.loads(line)["doc_id"] for line in f if line.strip()}

    todo = [IMG_DIR / f"{d}.tif" for d in manifest["doc_id"]
            if d not in done and (IMG_DIR / f"{d}.tif").exists()]
    print(f"[3/4] OCR: {len(done)} done, {len(todo)} to go")

    with OCR_LOG.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            for row in ocr_images(todo[i:i + batch_size]):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()                                # checkpoint each batch
            print(f"   {min(i + batch_size, len(todo))}/{len(todo)}")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")    # "，" -> ","
    return re.sub(r"\s+", " ", text).strip()            # one field per line -> one line


def build_parquet(manifest: pd.DataFrame, min_chars: int) -> None:
    if not OCR_LOG.exists():
        sys.exit("No OCR results yet.")
    ocr = (pd.read_json(OCR_LOG, lines=True, dtype={"doc_id": str})
             .drop_duplicates("doc_id", keep="last")
             .drop(columns=["path"]))

    df = manifest.merge(ocr, on="doc_id", how="left")
    
    df["error"] = df["error"].astype("object")   # all-null column loads as float; allow strings
    df.loc[df["extractor"].isna(), "error"] = "missing_image"
    df["text_raw"] = df.pop("text").fillna("")
    df["text_clean"] = df["text_raw"].map(clean)
    df["chars"] = df["text_clean"].str.len()
    df["usable"] = df["error"].isna() & (df["chars"] >= min_chars)

    cols = ["doc_id", "label_id", "label", "text_raw", "text_clean", "chars",
            "ocr_score", "is_scanned", "extractor", "error", "usable", "tar_key"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_parquet(OUT, index=False)

    print(f"[4/4] wrote {OUT} ({len(df)} rows, {int(df['usable'].sum())} usable)")
    print(df.groupby("label").agg(docs=("doc_id", "size"),
                                  usable=("usable", "sum"),
                                  mean_ocr=("ocr_score", "mean"),
                                  mean_chars=("chars", "mean")).round(3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--min-chars", type=int, default=30)
    args = ap.parse_args()

    manifest = build_manifest(args.per_class, args.seed)
    extract_images(manifest)
    run_ocr(manifest, args.batch_size)
    build_parquet(manifest, args.min_chars)


if __name__ == "__main__":
    main()