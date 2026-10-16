import json
import subprocess
from pathlib import Path

BACKEND_PY = Path(r"C:\Users\ASUS\Documents\GitHub\Intelligent-Document-Search\.venv\Scripts\python.exe")
RUNNER = (Path(__file__).resolve().parent / "intelligent_document_finetuning" / "tools" / "backend_ocr_runner.py")


def ocr_images(paths: list[str | Path]) -> list[dict]:
    stdin = "\n".join(str(Path(p).resolve()) for p in paths)
    proc = subprocess.run([str(BACKEND_PY), str(RUNNER)], input=stdin,
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"OCR runner failed:\n{proc.stderr}")
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]