# import asyncio
# import sys
# from pathlib import Path

# BACKEND = Path(r"C:\Users\ASUS\Documents\GitHub\Intelligent-Document-Search")
# sys.path.insert(0, str(BACKEND))



import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so "src" imports
from src.ocr_client import ocr_images

row = ocr_images(["data/raw/rvl_cdip/samples/80035521.tif"])[0]
print(f"doc_id={row['doc_id']} is_scanned={row['is_scanned']} "
      f"ocr_score={row['ocr_score']:.3f} chars={len(row['text'])} error={row['error']}")
print(row["text"][:500])