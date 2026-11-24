# intelligent-document-finetuning

ML training repo for the thesis *AI-Powered Document Archiving and Semantic Retrieval System*
(Provincial Accounting Office, Capitol Cagayan). It trains the models that the archiving agent
in **Intelligent-Document-Search** will call as tools:

- **Topic modeling (unsupervised):** BERTopic, to discover document groups and new types
- **Classification (supervised):** TF-IDF + Logistic Regression / LinearSVC baselines, SetFit main model

See `docs/ARCHITECTURE.pdf` for a full explanation of the structure.

## Folder map

| folder | what goes here | rule |
|---|---|---|
| `configs/` | settings for each experiment (YAML) | no code |
| `data/` | raw -> interim -> processed datasets | never committed; regenerate with scripts |
| `notebooks/` | exploration, numbered in order | try ideas here, then move finished code to `src/` |
| `scripts/` | small entry points you run from the terminal | grouped by task; they import from `src/` |
| `src/intelligent_document_finetuning/` | reusable code (the package) | never imports from `scripts/` |
| `models/` | trained models and run outputs | never committed |
| `notes/` | experiment logs and findings for the thesis | one file per experiment family |
| `tests/` | automated tests | fast, no private data required |
| `docs/` | architecture guide | |

## How to run (from the repo root)

```bash
# RVL-CDIP pipeline
python scripts/rvl_cdip/01_download_rvl_cdip.py
python scripts/rvl_cdip/02_prepare_rvl_cdip_corpus.py --per-class 100

# BERTopic runs
python scripts/topic_modeling/fit_bertopic.py --run-name v1_default
python scripts/topic_modeling/fit_bertopic.py --run-name v2_mcs10 --min-cluster-size 10
python scripts/topic_modeling/fit_bertopic.py --run-name v3_mcs25 --min-cluster-size 25

# PAO classification (after running notebooks/01_pao_metadata_preparation.ipynb -> pao_metadata_v2.parquet)
python scripts/classification/train_setfit.py --run-name setfit_v1_fewshot16 --samples-per-class 16 --batch-size 16 --num-iterations 20
python scripts/classification/train_setfit.py --run-name setfit_v1_iter2 --batch-size 32 --num-iterations 2

# Confidence threshold calibration for the archiving agent's LLM fallback
python scripts/classification/calibrate_threshold.py --model tfidf
python scripts/classification/calibrate_threshold.py --oof-parquet models/classification/setfit_v1_iter2/oof_predictions.parquet

# Real scanned PAO PDF OCR extraction, few-shot training, and HITL feedback retraining
python scripts/pao_ocr/01_extract_and_eval_real_ocr.py
python scripts/pao_ocr/02_train_real_ocr.py --run-name real_ocr_v1
python scripts/pao_ocr/02_train_real_ocr.py --run-name real_ocr_v2_hitl --feedback-file data/interim/hitl_feedback.csv --champion-run models/classification/real_ocr_v1/run.json

# Run unit tests
python -m unittest discover -s tests -v

# OCR smoke test (needs the Intelligent-Document-Search venv)
python scripts/debug/test_ocr_one.py
```

PAO exploration notebooks live in `notebooks/` (`01_pao_metadata_preparation.ipynb`, `02_pao_baseline_tfidf_logreg.ipynb`).
