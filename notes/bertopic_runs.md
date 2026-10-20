# BERTopic runs — RVL-CDIP v1

## Dataset
- Source: RVL-CDIP (aharley/rvl_cdip, train split index)
- Sample: 1,600 docs — 100 per class × 16 classes, seed 42
- OCR: RapidOCR via Intelligent-Document-Search backend (`_extract_text_fallback`)
- OCR errors: 0 / 1,600
- Usable docs (no error, ≥ 30 chars): **1,518**
- Weak classes: file_folder (40/100 usable — mostly blank covers),
  handwritten (mean OCR score 0.76 vs ~0.9 for others)

## Fixed pipeline settings
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim), cached
- UMAP: n_components=5, min_dist=0.0, metric=cosine, random_state=42
- HDBSCAN: metric=euclidean, cluster_selection_method=eom
- Vectorizer: English stopwords, min_df=2, ngram_range=(1,2), tokens ≥ 3 letters
- c-TF-IDF: reduce_frequent_words=True
- Labels are hidden from the model; used only for evaluation
- Only parameter varied in v1–v3: `min_cluster_size`

## Run log

| run | n_neighbors | min_cluster_size | min_samples | topics | outlier % | NMI all | ARI all | NMI clust. | ARI clust. | weighted purity | topics purity ≥0.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_default | 15 | 15 | 5 | 22 | 28.5 | 0.327 | 0.107 | 0.429 | 0.227 | 0.471 | 6 |
| v2_mcs10 | 15 | 10 | 5 | 36 | **27.1** | **0.345** | 0.103 | **0.436** | **0.229** | **0.497** | **10** |
| v3_mcs25 | 15 | 25 | 5 | 12 | 31.2 | 0.286 | 0.104 | 0.392 | 0.226 | 0.424 | 3 |

Weighted purity = share of clustered (non-outlier) docs that belong to their topic's dominant class.

**Winner: v2_mcs10** — lowest outliers (27.1%), highest NMI clustered (0.436),
highest weighted purity (0.497), most clean topics (10 with purity ≥ 0.7).
Trade-off: 36 topics (more granular; can be merged later with `reduce_topics()`).

**Sensitivity:** NMI clustered stays within 0.39–0.44 across min_cluster_size
10 / 15 / 25 → the ceiling is a property of the data (subject vs layout),
not the hyperparameters.

**Reproducibility:** v1_default run 3 times with identical results (seed 42).

## Run notes

### v1_default (min_cluster_size = 15) — baseline
- 22 topics, 433 outliers.
- One 261-doc catch-all topic about smoking (`0_smoking_said_nicotine_smokers`).
- OCR-junk topic `3_tif_lal_het_van` (94 docs).

### v2_mcs10 (min_cluster_size = 10) — winner
- 36 topics, 411 outliers (−22 vs baseline).
- The smoking catch-all split into subject topics: nicotine/asthma (5),
  advertising bans (6), taxes (7), smoking/weight (11), lung cancer (15).
  Still split by **subject**, not document type.
- Invoice topic grew to 192 docs and absorbed budgets
  (61 invoice, 42 budget, 18 form, 16 letter) → a general "money documents" topic.
- OCR-junk topic unchanged (`2_tif_lal_het_van`, 94 docs).
- 12 small topics of 10–16 docs (topics 24–35).

### v3_mcs25 (min_cluster_size = 25) — worst
- 12 topics, 473 outliers (+40 vs baseline).
- Clean small topics (filter specs, ad-spend budget, fax sheets) merged into
  larger topics or became outliers.
- OCR-junk topic grew to 158 docs (`1_telefax_van_deaths_datum`),
  merging junk pages with fax sheets.
- Lower on every metric.

## v1_default — topic interpretation

Purity = share of a topic's docs that belong to its most common true class.

| topic | size | keywords | my reading | dominant class | purity |
|---|---|---|---|---|---|
| -1 | 433 | indoor, morris, philip, kent | outliers (no group) | form | 0.10 |
| 0 | 261 | smoking, said, nicotine, smokers | catch-all: anything *about* smoking | news_article | 0.27 |
| 1 | 139 | university, cells, medicine, professor | academic CVs / research | resume | 0.60 |
| 2 | 118 | invoice, check, vendor, voucher | invoices & payments | invoice | 0.43 |
| 3 | 94 | tif, lal, het, van | **OCR junk** (handwritten/blank pages) | handwritten | 0.48 |
| 4 | 48 | sent, subject, bcc, original message | emails | email | 0.75 |
| 5 | 39 | vapor, chemical, hazard, safety | safety data sheets | specification | 0.56 |
| 6 | 38 | cig, supplier, average, blend | cigarette spec reports | memo | 0.29 |
| 7 | 30 | tipping, paper, adhesive, perf | material specifications | specification | 0.80 |
| 8 | 30 | records, retention, destruction | records-management policy | memo | 0.27 |
| 9 | 27 | grants, council, tobacco research | research grant letters | letter | 0.37 |
| 10 | 27 | coupon, photography, cost, prizes | promotion budgets | budget | 0.33 |
| 11 | 27 | leaf, cured, bright, burley | leaf tobacco specs | specification | 0.33 |
| 12 | 25 | terminate, ask, age, callback | survey scripts | questionnaire | 0.92 |
| 13 | 24 | grams, filter, rods, plug | filter specifications | specification | 0.83 |
| 14 | 23 | deaths, boe, nre, brans | partly OCR noise | handwritten | 0.52 |
| 15 | 20 | quarter, est, gross, estimate recap | ad-spend budgets | budget | 0.70 |
| 16 | 20 | marlboro, racing, tennis, sports | sports sponsorship ads | advertisement | 0.30 |
| 17 | 20 | fax, transmission, sender | fax cover sheets | form | 0.65 |
| 18 | 20 | camel, raleigh, milder | brand advertisements | advertisement | 0.55 |
| 19 | 19 | laboratories, hepatic, rats, enzymes | lab toxicology reports | scientific_report | 0.42 |
| 20 | 19 | cigarettes, read list, respondent | consumer questionnaires | questionnaire | 0.95 |
| 21 | 17 | nnk, nitric oxide, mutagenic, assay | chemistry research | scientific_report | 0.41 |

## Per-class recovery (v1_default)

| class | outlier % | share in best topic | verdict |
|---|---|---|---|
| resume | 7 | 0.84 | recovered well |
| news_article | 14 | 0.72 | recovered (but inside catch-all topic 0) |
| invoice | 28 | 0.52 | mostly recovered |
| handwritten | 23 | 0.47 | grouped by OCR noise, not meaning |
| specification | 12 | 0.24 | split into several subject topics |
| form | 42 | 0.13 | not recovered |
| memo | 42 | 0.13 | not recovered |
| letter | 40 | 0.13 | not recovered |

## Observations
1. min_cluster_size = 10 gives the best balance; 25 is too coarse.
2. Outliers stay at 27–31% in all runs — memo, letter and form are layout-defined
   and are not rescued by any cluster size.
3. Clean topics (purity ≥ 0.7): questionnaire, specification (filter / tipping),
   email, budget (ad-spend).
4. A large subject-based topic (smoking) mixes news, memos, letters,
   presentations and reports; smaller cluster sizes split it by sub-subject only.
5. The OCR-junk topic (handwritten / blank pages) persists in every run —
   a data-quality issue, not a tuning issue.
6. Topic keywords polluted by company names (philip, morris, lorillard, kent)
   and OCR fragments (tif, lal, het, boe, nre) → fix with custom stopwords (v4_clean_vocab).

## Interpretation (draft for thesis)
BERTopic groups documents by **content subject**, not by **document layout**.
Because every RVL-CDIP document comes from the tobacco industry, types with
distinctive vocabulary (questionnaire, resume, email, invoice, fax,
specification) are recovered, while types that share subject matter and differ
mainly in layout (memo, letter, form) are merged into subject topics or left as
outliers. Varying min_cluster_size (10 / 15 / 25) changes NMI by at most 0.044,
confirming the limit comes from the data rather than the hyperparameters.
This supports the system design: BERTopic for content discovery and tagging,
plus a supervised classifier (SetFit) for document-type identification.

## Next
- [x] Run v2_mcs10 and v3_mcs25, fill in the table, pick a winner
- [ ] v4_clean_vocab: custom stopwords + min_df=3, min_cluster_size=10
- [ ] Topic interpretation table for v4 (final model)
- [ ] Fill in final findings
- [ ] Move to PAO corpus

## Appendix — raw run.json scores

```
v1_default: {'n_topics': 22, 'outlier_pct': 28.5, 'ari_all': 0.107, 'nmi_all': 0.327, 'ari_clustered': 0.227, 'nmi_clustered': 0.429}
v2_mcs10:   {'n_topics': 36, 'outlier_pct': 27.1, 'ari_all': 0.103, 'nmi_all': 0.345, 'ari_clustered': 0.229, 'nmi_clustered': 0.436}
v3_mcs25:   {'n_topics': 12, 'outlier_pct': 31.2, 'ari_all': 0.104, 'nmi_all': 0.286, 'ari_clustered': 0.226, 'nmi_clustered': 0.392}
```

### v4_clean_vocab (min_cluster_size = 10, custom stopwords, min_df = 3) — FINAL
- Same clustering as v2_mcs10 (identical scores): stopwords and min_df only
  affect topic naming (c-TF-IDF), which runs after HDBSCAN.
- Cleaner names: removed company names (kent, lorillard) and some OCR
  fragments (inh, boe, brans). Invoice topic now `invoice_sales_check_carton`.
- Topic 2 (94 docs) still has junk keywords (`vel_thak_seton_een`) because its
  documents are unreadable OCR. Labelled as **OCR-noise topic**; not fixed with
  more stopwords. In production this can flag bad scans for review.