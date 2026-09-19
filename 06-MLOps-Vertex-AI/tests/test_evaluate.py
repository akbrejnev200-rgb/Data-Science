import numpy as np

from aml_detection.evaluate import calibrate_threshold, evaluate_final


class _ScoreIsFeatureModel:
    """predict_proba renvoie directement la feature comme probabilité."""

    def predict_proba(self, X):
        X = np.asarray(X)
        p1 = X[:, 0]
        return np.column_stack([1 - p1, p1])


class _NeverSuspectModel:
    def predict_proba(self, X):
        n = len(np.asarray(X))
        return np.column_stack([np.ones(n), np.zeros(n)])


def _synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    proba = rng.uniform(0, 1, n)
    y = (proba + rng.normal(0, 0.15, n) > 0.7).astype(int)
    return proba.reshape(-1, 1), y


def test_calibrate_threshold_reaches_target_recall_on_validation():
    X, y = _synthetic()
    threshold, table = calibrate_threshold(_ScoreIsFeatureModel(), X, y, target_recall=0.8)

    y_proba = _ScoreIsFeatureModel().predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    recall = ((y_pred == 1) & (y == 1)).sum() / y.sum()

    assert recall >= 0.8 - 1e-9
    assert set(table.keys()) == {0.5, 0.6, 0.7, 0.8, 0.9}


def test_calibrate_threshold_does_not_crash_when_target_unreachable():
    # Un modèle qui ne prédit jamais "suspect" ne peut jamais atteindre un rappel > 0.
    X, y = _synthetic()
    threshold, _ = calibrate_threshold(_NeverSuspectModel(), X, y, target_recall=0.8)
    assert threshold is not None


def test_evaluate_final_returns_expected_keys():
    X, y = _synthetic()
    result = evaluate_final(_ScoreIsFeatureModel(), X, y, threshold=0.5)

    assert set(result) == {"y_proba", "y_pred", "report", "confusion_matrix", "auc_roc", "auc_pr"}
    assert 0.0 <= result["auc_roc"] <= 1.0
    assert 0.0 <= result["auc_pr"] <= 1.0
