"""Minimal REST demo for the fake-news detector.

Requires the optional ``api`` extra::

    pip install flask
    python -m fakenews.cli train          # produce models/fakenews_pipeline.joblib
    python app/api.py                     # serves on http://127.0.0.1:5000

Endpoints
---------
GET  /health                 -> {"status": "ok"}
POST /predict  {"text": ...}  -> {"label", "is_fake", "confidence", "features"}
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fakenews.detect import FakeNewsDetector  # noqa: E402


def create_app():
    try:
        from flask import Flask, jsonify, request
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("Flask is required: pip install flask") from exc

    app = Flask(__name__)

    try:
        detector = FakeNewsDetector.load()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
        if not text.strip():
            return jsonify(error="Provide a non-empty 'text' field."), 400

        result = detector.predict(text)
        return jsonify(
            label=result.label,
            is_fake=result.is_fake,
            confidence=round(result.confidence, 4),
            features=[
                {"name": name, "contribution": round(weight, 4)}
                for name, weight in detector.explain(text)
            ],
        )

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000)
