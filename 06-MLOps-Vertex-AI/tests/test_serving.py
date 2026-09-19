"""Tests de l'API de démo (serving/app.py), sans appel réseau : le modèle est
remplacé par un faux, injecté directement dans l'état de l'app."""

import numpy as np
from fastapi.testclient import TestClient

from aml_detection.features import FEATURES
from serving.app import _state, app


class _FakeModel:
    """La probabilité "suspect" est simplement la 1ère feature reçue."""

    def predict_proba(self, X):
        X = np.asarray(X)
        p1 = np.clip(X[:, 0], 0, 1)
        return np.column_stack([1 - p1, p1])


def _client(threshold=0.5):
    _state["model"] = _FakeModel()
    _state["threshold"] = threshold
    _state["model_name"] = "FakeModel"
    return TestClient(app)


def _payload(**overrides):
    payload = {name: 0.0 for name in FEATURES}
    payload.update(overrides)
    return payload


def test_health_reports_model_loaded():
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_predict_flags_suspect_above_threshold():
    resp = _client(threshold=0.5).post("/predict", json=_payload(**{FEATURES[0]: 0.9}))
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "suspect"
    assert body["probability"] == 0.9
    assert body["threshold"] == 0.5


def test_predict_flags_normal_below_threshold():
    resp = _client(threshold=0.5).post("/predict", json=_payload(**{FEATURES[0]: 0.1}))
    assert resp.json()["verdict"] == "normal"


def test_index_page_serves_html():
    resp = _client().get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Détection AML" in resp.text
