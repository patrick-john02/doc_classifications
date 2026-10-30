# Classification runs - PAO metadata

Copied from the outputs in notebooks/02_pao_baseline_tfidf_logreg.ipynb.

## Dataset
- Source: PAO logbook Excel, 4 sheets -> label = sheet name
- 3,650 rows after removing blank text: COMMUNICATION 2,285 / CLEARANCE 698 / LEAVES 653 / AOsMEMOs 14
- 359 duplicate texts, 0 label conflicts -> split with StratifiedGroupKFold (groups = document_text)

## Results (5-fold StratifiedGroupKFold, seed 42)

| model | mean accuracy | mean macro-F1 | AOsMEMOs F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.991 | 0.859 | 0.44 |
| TF-IDF + Linear SVM | 0.993 | 0.869 | 0.52 |
| SetFit | - | - | - |

## Open issues
- [ ] Finish manual label review (146 rows pending) - many memos/orders sit in COMMUNICATION
- [ ] Remove control numbers (AO-2026-..., MEMO-2026-...) from AOsMEMOs text (leakage)
- [ ] Check whether COMMUNICATION's TYPE column gives the label away
- [ ] Save the reviewed dataset to data/processed/pao_metadata_v1.parquet
- [ ] Run SetFit on the same folds
