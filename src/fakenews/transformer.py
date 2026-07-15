"""Fine-tuned transformer detector.

The TF-IDF + linear model in :mod:`fakenews.detect` is fast, interpretable and
a genuinely strong baseline. Its blind spot is *meaning*: it sees words, not
context, so paraphrased or subtly-worded misinformation that avoids the obvious
clickbait vocabulary can slip past it.

A pretrained transformer (BERT/DistilBERT) fixes that. It arrives already
knowing English from massive self-supervised pretraining; we only **fine-tune**
it — nudge its weights on our labelled fake/real examples — so it learns the
task from a few hundred documents instead of millions. Each token attends to
every other token (self-attention), so the model represents a word *in context*
("shot" in "he was shot" vs. "a great shot"), which is exactly the contextual
signal bag-of-words throws away.

This module is an **optional extra**: it needs ``torch`` and ``transformers``
(`pip install "fakenews[transformer]"`). Import stays cheap — the heavy
dependencies are only imported inside methods — so the rest of the package works
without them, and a helpful error is raised if you use this class without them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DEFAULT_TRANSFORMER_PATH, TransformerConfig
from .data import load_dataset
from .detect import Prediction
from .evaluate import EvaluationResult, evaluate


def _require_backend():
    """Import torch + transformers, or explain how to get them."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "The transformer detector needs extra dependencies. Install them with:\n"
            '    pip install "fakenews[transformer]"\n'
            "(this pulls in torch and transformers)."
        ) from exc
    import torch
    import transformers

    return torch, transformers


def _resolve_device(requested: str):
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerDetector:
    """Fine-tune and query a pretrained transformer for fake-news detection.

    The public interface mirrors :class:`fakenews.detect.FakeNewsDetector`
    (``fit`` / ``predict`` / ``predict_batch`` / ``save`` / ``load``) so the two
    are drop-in interchangeable.
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        self.config = config or TransformerConfig()
        self.model = None
        self.tokenizer = None
        self._device = None

    # -- internals ---------------------------------------------------------
    def _encode(self, texts: Sequence[str]):
        return self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

    # -- training ----------------------------------------------------------
    def fit(self, df: Optional[pd.DataFrame] = None) -> EvaluationResult:
        """Fine-tune on a ``text``/``label`` DataFrame; return held-out metrics."""
        torch, transformers = _require_backend()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if df is None:
            df = load_dataset()

        X = df["text"].astype(str).tolist()
        y = df["label"].astype(int).tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        torch.manual_seed(self.config.random_state)
        self._device = _resolve_device(self.config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, num_labels=2
        ).to(self._device)

        enc = self._encode(X_train)
        dataset = torch.utils.data.TensorDataset(
            enc["input_ids"], enc["attention_mask"], torch.tensor(y_train)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        total_steps = max(1, len(loader) * self.config.epochs)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.config.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )

        self.model.train()
        for _epoch in range(self.config.epochs):
            for input_ids, attention_mask, labels in loader:
                optimizer.zero_grad()
                out = self.model(
                    input_ids=input_ids.to(self._device),
                    attention_mask=attention_mask.to(self._device),
                    labels=labels.to(self._device),
                )
                out.loss.backward()
                optimizer.step()
                scheduler.step()

        y_pred = [1 if p.is_fake else 0 for p in self.predict_batch(X_test)]
        return evaluate(y_test, y_pred)

    # -- inference ---------------------------------------------------------
    def _check_ready(self):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Detector is not trained. Call fit() or load() first."
            )

    def predict_batch(self, texts: Sequence[str]) -> List[Prediction]:
        import torch

        self._check_ready()
        self.model.eval()
        out: List[Prediction] = []
        bs = self.config.batch_size
        texts = list(texts)
        with torch.no_grad():
            for start in range(0, len(texts), bs):
                enc = self._encode(texts[start : start + bs])
                logits = self.model(
                    input_ids=enc["input_ids"].to(self._device),
                    attention_mask=enc["attention_mask"].to(self._device),
                ).logits
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                for p in probs:
                    is_fake = p >= 0.5
                    out.append(
                        Prediction(
                            label="fake" if is_fake else "real",
                            is_fake=bool(is_fake),
                            confidence=float(p if is_fake else 1.0 - p),
                        )
                    )
        return out

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    # -- persistence -------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        self._check_ready()
        path = Path(path or DEFAULT_TRANSFORMER_PATH)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # Persist the inference-relevant config alongside the weights.
        (path / "detector_config.json").write_text(
            json.dumps(
                {
                    "model_name": self.config.model_name,
                    "max_length": self.config.max_length,
                    "batch_size": self.config.batch_size,
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "TransformerDetector":
        torch, _ = _require_backend()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path or DEFAULT_TRANSFORMER_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"No fine-tuned transformer at {path}. "
                "Run `python -m fakenews.cli train --arch transformer` first."
            )

        meta = {}
        meta_path = path / "detector_config.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        config = TransformerConfig(
            model_name=meta.get("model_name", "distilbert-base-uncased"),
            max_length=meta.get("max_length", 128),
            batch_size=meta.get("batch_size", 16),
        )
        detector = cls(config)
        detector._device = _resolve_device(config.device)
        detector.tokenizer = AutoTokenizer.from_pretrained(path)
        detector.model = AutoModelForSequenceClassification.from_pretrained(path).to(
            detector._device
        )
        return detector
