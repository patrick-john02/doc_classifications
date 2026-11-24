"""Fast unit tests for PAO data loading, CV evaluation, thresholds, and classifiers.

Uses synthetic fixtures so tests run in ~2 seconds with zero dependency on private
PAO files or a GPU. Run with:
    python -m unittest discover -s tests -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from intelligent_document_finetuning.classification.setfit_model import SetFitClassifier
from intelligent_document_finetuning.classification.tfidf_logreg import TfidfLogRegClassifier
from intelligent_document_finetuning.data.feedback import (
    clean_rapidocr_artifacts,
    decide_promotion,
    merge_feedback_dataset,
)
from intelligent_document_finetuning.data.pao_metadata import load_pao_metadata
from intelligent_document_finetuning.evaluation.cv import evaluate_oof, make_cv_splits
from intelligent_document_finetuning.evaluation.thresholds import (
    evaluate_thresholds,
    pick_recommended_threshold,
)


def _make_synthetic_df() -> pd.DataFrame:
    """Create a small synthetic PAO-like dataset with duplicate groups and conflicts."""
    rows = []
    templates = {
        "AOsMEMOs": [
            "MEMORANDUM ORDER NO 10 POLICY ON BUDGET",
            "EXECUTIVE ORDER NO 02 CREATING COMMITTEE",
            "ADMINISTRATIVE ORDER NO 05 GUIDELINES",
            "MEMORANDUM ORDER NO 14 SUBMISSION OF REPORT",
            "EXECUTIVE ORDER NO 08 REORGANIZATION",
        ],
        "CLEARANCE": [
            "TRAVEL ABROAD JAPAN",
            "RETIREMENT CLEARANCE",
            "TRANSFER TO ANOTHER OFFICE",
            "RESIGNATION EFFECTIVE MAY",
            "TERMINAL PAY",
        ],
        "COMMUNICATION": [
            "INVITATION TO REGIONAL WORKSHOP",
            "LETTER REQUEST FOR CERTIFICATION OF FUNDS",
            "NOTICE OF MEETING ON ANNUAL BUDGET",
            "REPLY TO AUDIT OBSERVATION MEMORANDUM",
            "TRANSMITTAL OF FINANCIAL STATEMENTS",
        ],
        "LEAVES": [
            "SICK LEAVE",
            "VACATION LEAVE",
            "COMPENSATORY TIME OFF",
            "WELLNESS LEAVE",
            "MANDATORY FORCED LEAVE",
        ],
    }
    idx = 0
    for label, texts in templates.items():
        for txt in texts:
            # Repeat each text twice so group_id leakage checks are genuinely exercised
            for _ in range(2):
                rows.append({
                    "row_id": f"{label}-{idx}",
                    "document_text": txt,
                    "label": label,
                    "review_status": "verified",
                    "group_id": txt,
                    "has_label_conflict": txt == "SICK LEAVE",
                })
                idx += 1
    return pd.DataFrame(rows)


class TestDataLoader(unittest.TestCase):
    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_pao_metadata(Path("nonexistent_pao_file.parquet"))

    def test_filters_domain_review_and_conflicts(self) -> None:
        df = _make_synthetic_df()
        df.loc[0, "review_status"] = "needs_domain_review"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sample.parquet"
            df.to_parquet(p, index=False)

            loaded_default = load_pao_metadata(p)
            self.assertEqual(len(loaded_default), len(df) - 1)
            self.assertNotIn("needs_domain_review", set(loaded_default["review_status"]))

            loaded_no_conflict = load_pao_metadata(p, exclude_conflicts=True)
            self.assertEqual(int(loaded_no_conflict["has_label_conflict"].sum()), 0)

    def test_missing_required_columns_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.parquet"
            pd.DataFrame({"text": ["hello"]}).to_parquet(p, index=False)
            with self.assertRaises(ValueError):
                load_pao_metadata(p)


class TestCVAndMetrics(unittest.TestCase):
    def test_make_cv_splits_zero_leakage_and_full_partition(self) -> None:
        df = _make_synthetic_df()
        splits = make_cv_splits(df, n_splits=5, seed=42)
        self.assertEqual(len(splits), 5)

        val_all = np.concatenate([va for _, va in splits])
        self.assertEqual(len(val_all), len(df))
        self.assertEqual(len(set(val_all)), len(df))

        for tr, va in splits:
            overlap = set(df["group_id"].iloc[tr]).intersection(set(df["group_id"].iloc[va]))
            self.assertEqual(len(overlap), 0)

    def test_evaluate_oof_with_conflicts_and_custom_index(self) -> None:
        y_true = pd.Series(["A", "A", "B", "B"], index=[10, 20, 30, 40])
        y_pred = np.array(["A", "B", "B", "B"])
        has_conflict = pd.Series([False, True, False, False], index=[10, 20, 30, 40])

        res = evaluate_oof(y_true, y_pred, has_conflict=has_conflict)
        self.assertAlmostEqual(float(res["accuracy_all"]), 0.75, places=4)
        self.assertAlmostEqual(float(res["accuracy_no_conflict"]), 1.0, places=4)
        self.assertEqual(res["confusion_matrix"].shape, (2, 2))
        self.assertIn("A", res["per_class_f1"])


class TestThresholds(unittest.TestCase):
    def test_evaluate_thresholds_and_recommendation(self) -> None:
        y_true = np.array(["A", "A", "B", "B"])
        y_pred = np.array(["A", "B", "B", "B"])
        conf = np.array([0.95, 0.55, 0.85, 0.72])

        table = evaluate_thresholds(y_true, y_pred, conf, thresholds=(0.50, 0.70, 0.90))
        self.assertEqual(list(table["threshold"]), [0.5, 0.7, 0.9])
        # At t=0.70: the single wrong pred (conf=0.55) is caught -> 100% errors caught, 100% auto acc
        row_70 = table[table["threshold"] == 0.7].iloc[0]
        self.assertEqual(row_70["coverage_pct"], 75.0)
        self.assertEqual(row_70["auto_accuracy"], 1.0)
        self.assertEqual(row_70["errors_caught_pct"], 100.0)

        rec = pick_recommended_threshold(table, min_auto_accuracy=0.95)
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec["threshold"], 0.7)


class TestClassifiers(unittest.TestCase):
    def test_tfidf_fit_predict_agent_json_and_save_load(self) -> None:
        df = _make_synthetic_df()
        for use_char in (True, False):
            clf = TfidfLogRegClassifier(use_char_ngrams=use_char, min_df=1).fit(
                df["document_text"], df["label"]
            )
            sample_texts = ["MEMORANDUM ORDER NO 10 POLICY ON BUDGET", "TRAVEL ABROAD JAPAN"]
            preds = clf.predict(sample_texts)
            probs = clf.predict_proba(sample_texts)
            agent_out = clf.predict_with_confidence(sample_texts)

            self.assertEqual(preds.shape, (2,))
            self.assertEqual(probs.shape, (2, 4))
            np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), atol=1e-6)
            # Must be cleanly JSON-serializable for agent tool responses
            serialized = json.dumps(agent_out)
            self.assertIn("confidence", serialized)

            with tempfile.TemporaryDirectory() as td:
                saved = clf.save(Path(td) / "clf.joblib")
                reloaded = TfidfLogRegClassifier.load(saved)
                self.assertEqual(reloaded.predict_with_confidence(sample_texts), agent_out)

    def test_setfit_unfitted_guards(self) -> None:
        clf = SetFitClassifier()
        with self.assertRaises(RuntimeError):
            _ = clf.classes_
        with self.assertRaises(RuntimeError):
            clf.predict(["hello"])
        with self.assertRaises(RuntimeError):
            clf.predict_proba(["hello"])


class TestHITLFeedbackAndPromotion(unittest.TestCase):
    def test_clean_rapidocr_artifacts(self) -> None:
        raw = "OFFICE | [?CERTIFICATION]  THIS IS   TO CERTIFY"
        self.assertEqual(clean_rapidocr_artifacts(raw), "OFFICE CERTIFICATION THIS IS TO CERTIFY")

    def test_merge_feedback_dedups_and_enforces_min_class_samples(self) -> None:
        base = pd.DataFrame({
            "doc_id": [f"c-{i}" for i in range(8)] + [f"l-{i}" for i in range(8)],
            "filename": [f"f-{i}.pdf" for i in range(16)],
            "label": ["CLEARANCE"] * 8 + ["LEAVES"] * 8,
            "text_page1": ["CERTIFICATION TEXT"] * 8 + ["APPLICATION FOR LEAVE"] * 8,
        })
        # Override c-0 -> LEAVES, and add only 2 AOsMEMOs rows (below min_class_samples=8)
        fb = pd.DataFrame({
            "doc_id": ["c-0", "ao-1", "ao-2"],
            "ocr_text_page1": ["APPLICATION FOR LEAVE", "MEMORANDUM ORDER 1", "EXECUTIVE ORDER 2"],
            "human_label": ["LEAVES", "AOsMEMOs", "AOsMEMOs"],
        })
        merged, stats = merge_feedback_dataset(base, fb, min_class_samples=7)
        self.assertEqual(stats["overridden_doc_ids"], 1)
        self.assertIn("AOsMEMOs", stats["skipped_classes"])
        self.assertEqual(stats["skipped_classes"]["AOsMEMOs"], 2)
        self.assertEqual(merged.set_index("doc_id").loc["c-0", "label"], "LEAVES")

    def test_champion_vs_challenger_gate(self) -> None:
        champ = {
            "best_model": "m1",
            "results": [{
                "model": "m1",
                "macro_f1": 0.96,
                "per_class_f1": {"CLEARANCE": 0.98, "COMMUNICATION": 0.96, "LEAVES": 0.94},
            }],
        }
        # 1) Same classes, better F1 -> PROMOTE
        better = {
            "macro_f1": 0.97,
            "per_class_f1": {"CLEARANCE": 0.98, "COMMUNICATION": 0.97, "LEAVES": 0.96},
        }
        self.assertEqual(decide_promotion(better, champ)["decision"], "PROMOTE")

        # 2) Same classes, regression -> HOLD
        worse = {
            "macro_f1": 0.91,
            "per_class_f1": {"CLEARANCE": 0.91, "COMMUNICATION": 0.91, "LEAVES": 0.91},
        }
        self.assertEqual(decide_promotion(worse, champ)["decision"], "HOLD")

        # 3) New class AOsMEMOs unlocked with strong F1 -> PROMOTE
        unlocked = {
            "macro_f1": 0.94,
            "per_class_f1": {"AOsMEMOs": 0.88, "CLEARANCE": 0.97, "COMMUNICATION": 0.96, "LEAVES": 0.95},
        }
        self.assertEqual(decide_promotion(unlocked, champ)["decision"], "PROMOTE")

        # 4) New class AOsMEMOs too weak -> HOLD
        weak_new = {
            "macro_f1": 0.88,
            "per_class_f1": {"AOsMEMOs": 0.55, "CLEARANCE": 0.98, "COMMUNICATION": 0.97, "LEAVES": 0.96},
        }
        self.assertEqual(decide_promotion(weak_new, champ)["decision"], "HOLD")


if __name__ == "__main__":
    unittest.main()
