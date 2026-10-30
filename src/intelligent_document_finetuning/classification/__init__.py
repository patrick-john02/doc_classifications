"""Supervised document-type classification.

Planned:
    tfidf_logreg.py  baseline (currently in notebooks/02_pao_baseline_tfidf_logreg.ipynb)
    setfit_model.py  few-shot SetFit main model
Both should expose the same interface: fit(texts, labels) and predict_proba(texts),
so the agent can call either one as a tool.
"""
