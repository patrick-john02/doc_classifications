"""TF-IDF + Logistic Regression document classifier with a shared agent-ready interface.
This will be also a tool for the Document Archiving System Agents. 

Why this file exists:
- Both TF-IDF + LogReg (baseline) and SetFit (main model) will share the exact
  same methods: fit(), predict(), predict_proba(), predict_with_confidence(),
  save(), and load().
- That way, Intelligent-Document-Search can swap models without changing agent code.
"""

from pathlib import Path
from typing import Iterable, Sequence, cast
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import TransformerMixin


class TfidfLogRegClassifier:
    """Supervised text classifier using TF-IDF features + balanced Logistic Regression."""

    def __init__(
        self,
        use_char_ngrams: bool = True,
        min_df: int = 2,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.use_char_ngrams = use_char_ngrams
        self.min_df = min_df
        self.max_iter = max_iter
        self.random_state = random_state
        self.pipeline: Pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        if self.use_char_ngrams:
            word_vectorizer = TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2), min_df=self.min_df
            )
            char_vectorizer = TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=self.min_df,
            )
            transformers = [("word", word_vectorizer), ("char", char_vectorizer)]
            vectorizer = FeatureUnion(
                cast(Sequence[tuple[str, TransformerMixin | Pipeline]], transformers)
            )
        else:
            vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=self.min_df)

        clf = LogisticRegression(
            max_iter=self.max_iter,
            class_weight="balanced",
            random_state=self.random_state,
        )
        return Pipeline([("features", vectorizer), ("clf", clf)])

    @property
    def classes_(self) -> np.ndarray:
        """Ordered class names learned during fit()."""
        return self.pipeline.classes_

    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> "TfidfLogRegClassifier":
        """Train the TF-IDF + Logistic Regression pipeline on (texts, labels)."""

        self.pipeline.fit(list(texts), list(labels))

        return self
    

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        """Predict the top class label for each document text."""


        pred = self.pipeline.predict(list(texts))

        if not isinstance(pred, np.ndarray):
            raise TypeError("Expected NumPy array from pipeline.predict()")
        
        return pred


    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        """Return class probability matrix of shape (n_samples, n_classes)."""

        predict_probability = self.pipeline.predict_proba(list(texts))

        return predict_probability

    def predict_with_confidence(self, texts: Iterable[str]) -> list[dict]:
        """Agent-friendly output: label, top confidence score, and full probability map.

        Example item returned for one document:
            {
                "label": "AOsMEMOs",
                "confidence": 0.9142,
                "probs": {"AOsMEMOs": 0.9142, "CLEARANCE": 0.0101, ...},
            }
        """

        probability_matrix = self.predict_proba(texts)
        class_names=list(self.classes_)

        result=[]

        for row_probs in probability_matrix:
            best_index = int(np.argmax(row_probs))
            prediction={
                "label":str(class_names[best_index]),
                "confidence":round(float(row_probs[best_index]),4),
                "probs":{
                    str(c):round(float(p),4)
                    for c, p in zip(class_names, row_probs)
                },
            }
            result.append(prediction)

        return result

    def save(self, path: Path | str) -> Path:
        """Persist this trained classifier to disk via joblib."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)
        return out_path

    @classmethod
    def load(cls, path: Path | str) -> "TfidfLogRegClassifier":
        """Load a previously saved classifier from disk."""
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
