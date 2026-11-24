# Classification runs - PAO metadata

## v0 (Archived baseline — `notebooks/archive/old_02_pao_baseline_tfidf_logreg.ipynb`)
- **Dataset:** Raw 4 sheets with label = sheet name (`3,650` rows: `COMMUNICATION 2,285 / CLEARANCE 698 / LEAVES 653 / AOsMEMOs 14`).
- **Known issues (why accuracy looked artificially high at 0.991):**
  - Included logbook-column leakage in `document_text`: `COMMUNICATION` included `TYPE` (`IN`/`OUT`), `CLEARANCE` included `REMARKS` (`RELEASED`) and `C/O`, `AOsMEMOs` included `AO/MEMO/EO #` (`EO-2026-01`, `AO-2026-02`), and `LEAVES` included `DAY/S APPLIED`.
  - 13 of 14 `AOsMEMOs` rows were duplicated in `COMMUNICATION`, and ~133 orders/memos sat mislabeled in `COMMUNICATION`.

| model | mean accuracy | mean macro-F1 | AOsMEMOs F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression (v0, leaked) | 0.991 | 0.859 | 0.44 |
| TF-IDF + Linear SVM (v0, leaked) | 0.993 | 0.869 | 0.52 |

---

## v1 (Clean leak-free baseline — `notebooks/02_pao_baseline_tfidf_logreg.ipynb`)
- **Dataset:** `data/processed/pao_metadata_v1.parquet` (`3,619` rows after removing 10 blanks, 13 cross-sheet `AOsMEMOs`↔`COMMUNICATION` duplicates, and 12 `needs_domain_review` rows).
- **Class counts:** `COMMUNICATION 2,127 / CLEARANCE 693 / LEAVES 652 / AOsMEMOs 147`.
- **Text features:** Pure document text only (`SUBJECT`+`DESCRIPTION` for `COMMUNICATION`, `PURPOSE` for `CLEARANCE`, `SUBJECT` for `AOsMEMOs`, `TYPE OF LEAVE`+`REMARKS` for `LEAVES`). Zero control numbers, `TYPE`, `C/O`, or `REMARKS="RELEASED"`.
- **Validation:** 5-fold `StratifiedGroupKFold` (`groups = group_id`, `seed = 42`, verified 0 group overlap across all folds).
- **Run artifacts:** `models/classification/tfidf_logreg_v1/` (`run.json`, `tfidf_word_logreg.joblib`, `best_model.joblib`).

| model | mean accuracy | mean macro-F1 | AOsMEMOs F1 | CLEARANCE F1 | COMMUNICATION F1 | LEAVES F1 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF (word + char_wb) + Logistic Regression | **0.9104** (±0.0535) | **0.8536** (±0.0881) | 0.8860 | **0.9388** | **0.9378** | 0.7872 |
| TF-IDF (word) + Linear SVM | 0.8887 (±0.0949) | 0.8263 (±0.1396) | **0.9123** | 0.8627 | 0.9229 | **0.7947** |
| TF-IDF (word) + Logistic Regression | 0.8229 (±0.0762) | 0.7223 (±0.1007) | 0.9091 | 0.8379 | 0.8769 | 0.5168 |

### Key findings from v1
1. **`AOsMEMOs` F1 jumped from `0.44` → `0.9091` (word LogReg) / `0.9123` (LinearSVC)** after deduplicating cross-sheet twins and reviewing mislabeled orders/memos in `COMMUNICATION`.
2. **Why `LEAVES` recall drops under strict grouped CV (`0.365` for word-only LogReg, rising to `0.678` with char n-grams):** `LEAVES` has only 42 unique texts across 652 rows; when a single huge 1-word group like `COMPENSATORY` (132 rows) or `SPL` (109 rows) lands in the validation fold, it is completely out-of-vocabulary for word TF-IDF (all-zeros vector). Adding `char_wb (3, 5)` n-grams raises macro-F1 by **+0.131** (`0.7223` → `0.8536`, OOF macro-F1 `0.8874`).
3. **Model-assisted error analysis surfaced ~9 more missed `AOsMEMOs` in `COMMUNICATION`:** texts using `MEMO. ORDER NO.`, `MEMO NO.`, `ADMIN ORDER`, or `MEMORANDUM:` (e.g., `COMMUNICATION-834`, `967`, `1421`, `1557`, `1663`, `1752`, `1889`, `2001`).
4. **Identical-text metadata collisions (`354` rows across 5 groups):** `SICK LEAVE`, `TERMINAL`, `MATERNITY`, `STUDY LEAVE`, and `VACATION LEAVE` appear identically in both `CLEARANCE.PURPOSE` and `LEAVES.TYPE OF LEAVE`.

---

## v2 (Error-analysis cleaned baseline — `notebooks/02_pao_baseline_tfidf_logreg.ipynb`)
- **Dataset:** `data/processed/pao_metadata_v2.parquet` (`3,618` rows after v2 review pass; `13` `needs_domain_review` rows exported to `data/interim/pao_ask_domain_expert.csv`).
- **What changed from v1 → v2:**
  - Expanded `ORDER_PATTERN` in `01_pao_metadata_preparation.ipynb` to catch abbreviation/punctuation variants surfaced by v1 error analysis (`MEMO. ORDER`, `MEMO NO.`, `ADMIN ORDER`, `^MEMORANDUM`, `^MEMO`).
  - Reviewed **21 newly caught rows**: `15` verified `AOsMEMOs` (`COMMUNICATION-312`, `834`, `920`, `967`, `1421`, `1557`, `1640`, `1663`, `1723`, `1752`, `1753`, `1871`, `1889`, `2001`, `2080`), `5` verified `COMMUNICATION` (MOAs + external DILG memos/circulars), `1` `needs_domain_review` (`COMMUNICATION-1014`).
- **Class counts:** `COMMUNICATION 2,111 / CLEARANCE 693 / LEAVES 652 / AOsMEMOs 162`.
- **Validation:** 5-fold `StratifiedGroupKFold` (`groups = group_id`, `seed = 42`, verified 0 group overlap across all folds).
- **Run artifacts:** `models/classification/tfidf_logreg_v2/` (`run.json`, `tfidf_word_logreg.joblib`, `best_model.joblib`).

| model | mean accuracy | mean macro-F1 | AOsMEMOs F1 | CLEARANCE F1 | COMMUNICATION F1 | LEAVES F1 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF (word) + Linear SVM | **0.9576** (±0.0254) | **0.9291** (±0.0346) | 0.8961 | **0.9465** | **0.9756** | **0.9249** |
| TF-IDF (word + char_wb) + Logistic Regression | 0.9026 (±0.0468) | 0.8378 (±0.0879) | **0.9024** | 0.9349 | 0.9329 | 0.7371 |
| TF-IDF (word) + Logistic Regression | 0.8140 (±0.0769) | 0.7015 (±0.1142) | 0.8862 | 0.7905 | 0.8808 | 0.4911 |

### Key findings from v2
1. **`tfidf_word_char_logreg` `AOsMEMOs` F1 rose from `0.8860` → `0.9024`** after fixing the 15 missed `MEMO NO.` / `MEMO. ORDER` / `ADMIN ORDER` rows, and `tfidf_word_linsvc` reached **94.5% precision** on `AOsMEMOs` (only 8 false positives out of 2,111 `COMMUNICATION` rows) with **0.9358 OOF Macro-F1** (`0.9441` excluding the 354 `CLEARANCE`↔`LEAVES` identical-text conflict rows).
2. **High fold sensitivity on `LEAVES` (42 unique groups):** Shifting 16 rows changed how `StratifiedGroupKFold` distributed the largest 1-word `LEAVES` groups across the 5 folds, which moved `LinearSVC`'s `LEAVES` F1 from `0.795` → `0.925` (`std` dropped from `±0.140` → `±0.035`). Reporting `mean ± std` (and comparing SetFit on the *exact same* v2 folds) keeps evaluation honest.
3. **Remaining `AOsMEMOs` misses are mostly unnumbered memos** (`MEMO TO ...`, `MEMO RE: ...`, `MEMORANDUM: ...`) where TF-IDF relies heavily on `"ORDER NO."` tokens—a clear target for SetFit's semantic representations.

---

## SetFit & Semantic Embedding Runs (`pao_metadata_v2.parquet`, identical 5 folds via `src/.../evaluation/cv.py`)
- **Backbone:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (RTX 5060 GPU, AMP enabled).
- **Script:** `scripts/classification/train_setfit.py` using `SetFitClassifier` (`src/intelligent_document_finetuning/classification/setfit_model.py`).
- **Artifacts:** `models/classification/setfit_v1_fewshot16/` and `models/classification/setfit_v1_iter2/` (`run.json`, `oof_predictions.parquet`, `model/`).

| model | mean accuracy | mean macro-F1 | OOF macro-F1 | OOF macro-F1 (no conflict) | AOsMEMOs F1 | CLEARANCE F1 | COMMUNICATION F1 | LEAVES F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **SetFit Full-Data (`iter=2`, `bs=32`)** | **0.9725** (±0.0087) | **0.9532** (±0.0110) | **0.9548** | **0.9672** | **0.9217** | **0.9505** | **0.9880** | **0.9591** |
| TF-IDF (word) + LinearSVC | 0.9576 (±0.0254) | 0.9291 (±0.0346) | 0.9358 | 0.9441 | 0.8961 | 0.9465 | 0.9756 | 0.9249 |
| TF-IDF (word + `char_wb`) + LogReg | 0.9026 (±0.0468) | 0.8378 (±0.0879) | 0.8769 | 0.8678 | 0.9024 | 0.9349 | 0.9329 | 0.7371 |
| Frozen MiniLM + LogReg *(no fine-tuning)* | — | — | 0.8198 | 0.8216 | 0.6651 | 0.8259 | 0.9536 | 0.8346 |
| SetFit Few-Shot (`16/class`, `iter=20`) | 0.8447 (±0.0800) | 0.7613 (±0.1311) | 0.7730 | 0.8046 | 0.6808 | 0.6969 | 0.9377 | 0.7766 |
| TF-IDF (word) + LogReg *(config baseline)* | 0.8140 (±0.0769) | 0.7015 (±0.1142) | 0.7621 | 0.7502 | 0.8862 | 0.7905 | 0.8808 | 0.4911 |

### Key findings from SetFit runs
1. **SetFit (`setfit_v1_iter2`) beat every baseline on every single metric and every single class:** `0.9725` accuracy (`±0.0087`), `0.9532` mean macro-F1 (`±0.0110` — 3× tighter fold stability than LinearSVC's `±0.0346`), and **`0.9672` OOF macro-F1** excluding the 354 identical-text `CLEARANCE`↔`LEAVES` collisions.
2. **Why contrastive fine-tuning matters (Frozen MiniLM → SetFit):** Frozen MiniLM already lifted `LEAVES` F1 to `0.8346` via pretrained semantics, but collapsed on `AOsMEMOs` (`0.6651`) because generic embeddings group PGC memos and letters by topic. Contrastive fine-tuning pushed `AOsMEMOs` F1 up by **+0.2566** (`0.6651` → **`0.9217`**, 153/162 recalled = 94.4%) while also lifting `LEAVES` to **`0.9591`**.
3. **Few-shot ablation (`setfit_v1_fewshot16`, only 64 rows/fold):** Achieved **98.6% recall on `LEAVES` (643/652)** and **89.5% recall on `AOsMEMOs` (145/162)** with Fold 1 reaching `0.9508` macro-F1, demonstrating strong low-data potential when moving to real scanned OCR samples.

### Confidence Threshold Calibration (for Archiving Agent LLM Fallback)
Calibrated via `scripts/classification/calibrate_threshold.py` (`src/.../evaluation/thresholds.py`) on 5-fold out-of-fold probabilities (`TfidfLogRegClassifier`, word + `char_wb`, `n = 3,618`):

| threshold | auto-archive coverage | auto-archive accuracy | LLM fallback rate | errors safely caught |
|---:|---:|---:|---:|---:|
| `0.50` | 95.14% | 0.9236 | 4.86% | 25.07% |
| `0.70` | 80.10% | 0.9648 | 19.90% | 70.94% |
| `0.75` | 78.61% | 0.9698 | 21.39% | 75.50% |
| **`0.80`** *(>=97% target)* | **75.95%** | **0.9709** | **24.05%** | **77.21%** |
| `0.85` | 70.18% | 0.9783 | 29.82% | 84.33% |
| `0.95` | 26.67% | 0.9969 | 73.33% | 99.15% |

*(Excluding the 354 identical-text conflict rows, `T = 0.75` reaches `97.50%` auto-accuracy at `80.82%` coverage, catching `78.78%` of errors. `train_setfit.py` is now updated to save OOF `confidence` automatically for future SetFit calibration runs.)*

---

## Real Scanned PAO PDF Evaluation & Few-Shot Domain Adaptation (`n = 56` real PDFs)
- **Corpus:** `data/processed/pao_real_ocr_v1.parquet` (`COMMUNICATION 25 / LEAVES 16 / CLEARANCE 15`; `AOsMEMOs 0` scans currently available; mean RapidOCR confidence `0.948`, `0` extraction errors; median Page 1 words: `CLEARANCE 76`, `COMMUNICATION 71`, `LEAVES 168.5`).
- **Scripts:** `scripts/pao_ocr/01_extract_and_eval_real_ocr.py` (Stage 1 zero-shot transfer) and `scripts/pao_ocr/02_train_real_ocr.py` (Path B 5-fold `StratifiedKFold`, `seed = 42`, Page 1 OCR with `[?...]` and `|` artifacts cleaned).
- **Artifacts:** `models/classification/real_ocr_eval_v1.json` and `models/classification/real_ocr_v1/` (`run.json`, `oof_predictions.parquet`, `model.joblib`).

| model | setup | accuracy | macro-F1 | CLEARANCE F1 | COMMUNICATION F1 | LEAVES F1 |
|---|---|---:|---:|---:|---:|---:|
| Zero-Shot Metadata SetFit (`setfit_v1_iter2`) | Metadata → Real OCR (0 real scans) | 0.4464 *(25/56)* | 0.2058 | 0.0000 | 0.6173 | 0.0000 |
| Zero-Shot Metadata TF-IDF (`tfidf_v2_best`) | Metadata → Real OCR (0 real scans) | 0.5536 *(31/56)* | 0.3229 | 0.0000 | 0.7463 | 0.5455 |
| **TF-IDF (`word + char_wb`) on Real OCR** | 5-fold CV (~45 real scans/fold) | **0.9821** *(55/56)* | **0.9827** | **1.0000** | **0.9804** | **0.9677** |
| **Few-Shot SetFit (Fresh MiniLM)** | 5-fold CV (~45 real scans/fold, `iter=20`) | 0.9643 *(54/56)* | 0.9658 | **1.0000** | 0.9600 | 0.9375 |
| **Domain-Adapted SetFit (`setfit_v1_iter2` → OCR)** | 5-fold CV (~45 real scans/fold, `iter=20`) | 0.9643 *(54/56)* | 0.9652 | 0.9677 | 0.9600 | **0.9677** |

### Key findings from Real-OCR validation
1. **Severe metadata → real-OCR domain shift exposed & solved:** Metadata-only models collapsed zero-shot (`44.6%` SetFit, predicting `COMMUNICATION` on all 56/56 scans) because Excel `CLEARANCE`/`LEAVES` cells are 1–2 words while real forms are 75–170 words with RapidOCR fused tokens (`APPLICATIONFORLEAVE`, `CERTIFICATION THISISTO CERTIFYTHAT`). Training on just ~45 real scans/fold lifted accuracy by **+51.8 points** (`44.6%` → `96.4%` SetFit / `98.2%` TF-IDF char).
2. **Real OCR completely eliminates the Excel `CLEARANCE` ↔ `LEAVES` collision:** `CLEARANCE` jumped from `0/15` → **`15/15` (`1.000` F1)** and `LEAVES` jumped from `0/16` → **`15/16` (`0.9677` F1)**.
3. **The 1 single `LEAVES` error (`2024-12-03-LEAVE-OUT-01.pdf`) was genuinely a cover letter on Page 1:** Page 1 (`39` words) literally reads `"DEAR SIR: MAY I RESPECTFULLY TRANSMIT THE APPLICATION FOR LEAVE OF..."` with CSC Form 6 attached on Page 2 (`188` words total). All 3 models predicted `COMMUNICATION` on Page 1, and TF-IDF's confidence dropped to **`0.722`**, so a `T = 0.75` fallback threshold routes it straight to the LLM fallback (`55/55` = `100%` auto-archive accuracy on remaining docs!).
4. **Human-in-the-Loop (HITL) continual learning ready:** `src/.../data/feedback.py` + `02_train_real_ocr.py` support `--feedback-file`, `--min-class-samples 8`, and `--champion-run` gating so `Intelligent-Document-Search` can safely learn `AOsMEMOs` (and user corrections) from bulk-upload HITL edits.

---

## Open issues / Next steps
- [x] Remove 13 cross-sheet `AOsMEMOs`↔`COMMUNICATION` duplicates
- [x] Complete first-pass (`v1`) and error-driven (`v2`) label review (`167 verified`, `13 needs_domain_review` exported to `data/interim/pao_ask_domain_expert.csv`)
- [x] Remove logbook-column leakage (`CONTROL#`, `AO/MEMO/EO #`, `TYPE`, `C/O`, `REMARKS="RELEASED"`)
- [x] Export clean datasets to `data/processed/pao_metadata_v1.parquet` and `pao_metadata_v2.parquet`
- [x] Run clean 5-fold `StratifiedGroupKFold` TF-IDF + Logistic Regression / LinearSVC baselines (`v1` & `v2`)
- [x] Move data loading, CV evaluation, TF-IDF, SetFit, threshold calibration, and HITL feedback gating into `src/intelligent_document_finetuning/`
- [x] Run Frozen MiniLM, Few-Shot SetFit (`16/class`), and Full-Data SetFit (`iter=2`) on the identical v2 5-fold splits
- [x] Build threshold calibration module + calibrate TF-IDF baseline (`T = 0.80` for `>= 97%` auto-archive accuracy)
- [x] Validate on 56 real RapidOCR-scanned PAO PDFs (`44.6%` zero-shot → `98.2%` TF-IDF char / `96.4%` few-shot SetFit)
- [x] Add automated unit tests in `tests/` and clean up `README.md`
- [ ] Confirm the 13 `needs_domain_review` rows in `data/interim/pao_ask_domain_expert.csv` with PAO staff
- [ ] Collect `>= 8` real `AOsMEMOs` PDFs via PAO or HITL feedback in `Intelligent-Document-Search` and re-run `02_train_real_ocr.py --champion-run models/classification/real_ocr_v1/run.json`
- [ ] Wire `real_ocr_v1` + `classification_feedback` HITL table into `Intelligent-Document-Search`
