import asyncio
import contextlib
import json
import sys
from pathlib import Path

BACKEND = Path(r"C:\Users\ASUS\Documents\GitHub\Intelligent-Document-Search")
sys.path.insert(0, str(BACKEND))

out = sys.stdout
out.reconfigure(encoding="utf-8")

with contextlib.redirect_stdout(sys.stderr):          # keep extractor logs off stdout
    from rag.extractor import _extract_text_fallback  # RapidOCR route for images


async def run(paths: list[Path]) -> None:
    for p in paths:
        row = {"doc_id": p.stem, "path": str(p)}
        try:
            with contextlib.redirect_stdout(sys.stderr):
                # ".tiff" because the fallback doesn't recognise ".tif"
                r = await _extract_text_fallback(p.read_bytes(), p.stem + ".tiff")
            row.update(text=r.text, is_scanned=r.is_scanned, ocr_score=r.ocr_score,
                       extractor=getattr(r, "extractor", "rapidocr"), error=None)
        except Exception as e:
            row.update(text="", is_scanned=True, ocr_score=0.0, extractor="error", error=str(e))
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()


if __name__ == "__main__":
    paths = [Path(line.strip()) for line in sys.stdin if line.strip()]
    asyncio.run(run(paths))