"""
API de démo : score de risque AML pour un compte.

Charge le modèle (best_model.joblib) et le seuil (decision_threshold.joblib)
depuis l'artefact GCS du modèle enregistré dans le Vertex AI Model Registry
(``MODEL_ARTIFACT_URI``) — le code de service est découplé du modèle : on
change de version sans reconstruire l'image.
"""

import os
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage
from pydantic import create_model

from aml_detection.features import FEATURES

MODEL_ARTIFACT_URI = os.environ.get(
    "MODEL_ARTIFACT_URI",
    "gs://mlops-vertex-demo-aml/pipeline-root/69646548838/"
    "aml-training-20260911004231/train_6702289541920718848/model",
)
LOCAL_DIR = Path("/tmp/model")
INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

app = FastAPI(
    title="AML Suspect Scoring — Démo",
    description="Score de risque de blanchiment par compte, à partir du modèle "
    "enregistré dans le Vertex AI Model Registry.",
    version="1.0.0",
)

_state = {"model": None, "threshold": None, "model_name": None}

# Le corps de /predict : un champ float par feature, généré depuis la même
# liste que l'entraînement — impossible de désynchroniser les deux.
AccountFeatures = create_model(
    "AccountFeatures", **{name: (float, 0.0) for name in FEATURES}
)


def _download(gcs_uri: str, filename: str) -> Path:
    bucket_name, _, prefix = gcs_uri.removeprefix("gs://").partition("/")
    dest = LOCAL_DIR / filename
    storage.Client().bucket(bucket_name).blob(f"{prefix.rstrip('/')}/{filename}").download_to_filename(str(dest))
    return dest


@app.on_event("startup")
def load_model() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _download(MODEL_ARTIFACT_URI, "best_model.joblib")
    threshold_path = _download(MODEL_ARTIFACT_URI, "decision_threshold.joblib")
    _state["model"] = joblib.load(model_path)
    _state["threshold"] = float(joblib.load(threshold_path))
    _state["model_name"] = type(_state["model"]).__name__
    print(f"Modèle chargé depuis {MODEL_ARTIFACT_URI} — seuil = {_state['threshold']:.3f}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.post("/predict")
def predict(features: AccountFeatures) -> dict:
    x = [[getattr(features, name) for name in FEATURES]]
    proba = float(_state["model"].predict_proba(x)[0, 1])
    threshold = _state["threshold"]
    return {
        "probability": round(proba, 4),
        "threshold": round(threshold, 4),
        "verdict": "suspect" if proba >= threshold else "normal",
        "model": _state["model_name"],
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML
