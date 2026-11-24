"""SetFit document classifier sharing the exact same interface as TfidfLogRegClassifier.

Why this file exists:
- Main supervised model for the thesis (see configs/classification/setfit.yaml).
- Uses contrastive fine-tuning on a multilingual sentence-transformer backbone
  (paraphrase-multilingual-MiniLM-L12-v2) + a balanced LogisticRegression head.
- Exposes the EXACT same methods as TfidfLogRegClassifier:
    fit(), predict(), predict_proba(), predict_with_confidence(), save(), load()
  so Intelligent-Document-Search can swap TF-IDF -> SetFit with a 1-line change.
"""

from pathlib import Path
import tempfile
from typing import Iterable
import numpy as np
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

DEFAULT_BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SetFitClassifier:
    """Contrastive few-shot / fine-tuned sentence-transformer document classifier."""

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        batch_size: int = 16,
        num_epochs: int = 1,
        num_iterations: int | None = 2,
        max_steps: int = -1,
        seed: int = 42,
        use_amp: bool = True,
    ) -> None:
        self.base_model = base_model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_iterations = num_iterations
        self.max_steps = max_steps
        self.seed = seed
        self.use_amp = use_amp
        self.model: SetFitModel | None = None

    @property
    def classes_(self) -> np.ndarray:
        """Ordered class names learned during fit()."""
        if self.model is None or not self.model.labels:
            raise RuntimeError("Model must be fitted or loaded before accessing classes_.")
        return np.asarray(self.model.labels)

    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> "SetFitClassifier":
        """Contrastively fine-tune the sentence-transformer body and fit the LR head.

        SetFit training runs in 2 automatic phases inside Trainer.train():
            Phase 1: Generate positive/negative text pairs & fine-tune the MiniLM encoder
                     with CosineSimilarityLoss on GPU.
            Phase 2: Encode all training texts with the fine-tuned encoder & fit the
                     LogisticRegression classification head.
        """
        text_list = list(texts)
        label_list = list(labels)
        sorted_labels = sorted(set(label_list))
        label_to_id = {lbl: idx for idx, lbl in enumerate(sorted_labels)}

        train_dataset = Dataset.from_dict({
            "text": text_list,
            "label": [label_to_id[lbl] for lbl in label_list],
        })

        # When continuing from a previously saved SetFit folder (domain adaptation),
        # model_head.pkl already exists and SetFitModel.from_pretrained rejects head_params.
        has_saved_head = (Path(self.base_model) / "model_head.pkl").exists()
        load_kwargs: dict = {"labels": sorted_labels}
        if not has_saved_head:
            load_kwargs["head_params"] = {
                "class_weight": "balanced",
                "max_iter": 1000,
            }

        self.model = SetFitModel.from_pretrained(self.base_model, **load_kwargs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                output_dir=tmp_dir,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                num_iterations=self.num_iterations,
                max_steps=self.max_steps,
                seed=self.seed,
                use_amp=self.use_amp,
                report_to="none",
                save_strategy="no",
            )
            trainer = Trainer(model=self.model, args=args, train_dataset=train_dataset)
            trainer.train()

        return self

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        """Predict the top class label string for each document text."""
        if self.model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() or load() first.")
        text_list = list(texts)
        if not text_list:
            return np.array([], dtype=object)

        preds = self.model.predict(text_list, as_numpy=True)
        return np.asarray(preds, dtype=object)

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        """Return class probability matrix of shape (n_samples, n_classes)."""
        if self.model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() or load() first.")
        text_list = list(texts)
        if not text_list:
            return np.empty((0, len(self.classes_)), dtype=float)

        probs = self.model.predict_proba(text_list, as_numpy=True)
        return np.asarray(probs, dtype=float)

    def predict_with_confidence(self, texts: Iterable[str]) -> list[dict]:
        """Agent-friendly output (identical schema to TfidfLogRegClassifier)."""
        probs_matrix = self.predict_proba(texts)
        class_names = list(self.classes_)

        results: list[dict] = []
        for row_probs in probs_matrix:
            best_i = int(np.argmax(row_probs))
            results.append({
                "label": str(class_names[best_i]),
                "confidence": round(float(row_probs[best_i]), 4),
                "probs": {
                    str(c): round(float(p), 4)
                    for c, p in zip(class_names, row_probs)
                },
            })
        return results

    def save(self, path: Path | str) -> Path:
        """Save the fine-tuned SetFit model directory to disk."""
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted model.")
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out_dir))
        return out_dir

    @classmethod
    def load(cls, path: Path | str) -> "SetFitClassifier":
        """Load a saved SetFit model directory from disk."""
        wrapper = cls()
        wrapper.model = SetFitModel.from_pretrained(str(Path(path)))
        return wrapper
