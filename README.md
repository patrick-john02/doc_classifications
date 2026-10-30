# intelligent-document-finetuning

ML training repo for the thesis *AI-Powered Document Archiving and Semantic Retrieval System*
(Provincial Accounting Office, Capitol Cagayan). It trains the models that the archiving agent
in **Intelligent-Document-Search** will call as tools:

- **Topic modeling (unsupervised):** BERTopic, to discover document groups and new types
- **Classification (supervised):** TF-IDF + Logistic Regression baseline, SetFit main model

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
| `tests/` | automated tests | |
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

# OCR smoke test (needs the Intelligent-Document-Search venv)
python scripts/debug/test_ocr_one.py
```

PAO classification work is in `notebooks/` (open with Jupyter from the `notebooks/` folder).
