"""
Évaluation des modèles : métriques, calibration du seuil de décision.
"""

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve
)


def evaluate_isolation_forest(model, X_scaled, y):
    """Compare les anomalies détectées par Isolation Forest à la vérité terrain."""
    anomaly_pred = (model.predict(X_scaled) == -1).astype(int)
    report = classification_report(
        y, anomaly_pred, target_names=["normal", "suspect"],
        zero_division=0, output_dict=True,
    )
    detection_rate = (anomaly_pred[y == 1].sum() / y.sum()) * 100 if y.sum() > 0 else 0
    return anomaly_pred, report, detection_rate


def calibrate_threshold(model, X_val, y_val):
    """Trouve le seuil qui maximise le F1-score sur le jeu de validation."""
    val_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)

    f1_scores = np.divide(
        2 * precisions * recalls, precisions + recalls,
        out=np.zeros_like(precisions), where=(precisions + recalls) != 0
    )
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]

    threshold_table = {
        target_recall: {
            "threshold": thresholds[np.argmin(np.abs(recalls[:-1] - target_recall))],
            "precision": precisions[np.argmin(np.abs(recalls[:-1] - target_recall))],
            "recall": recalls[np.argmin(np.abs(recalls[:-1] - target_recall))],
        }
        for target_recall in [0.5, 0.6, 0.7, 0.8]
    }

    return best_threshold, threshold_table


def evaluate_final(model, X_test, y_test, threshold=0.5):
    """Évaluation finale sur le jeu de test, à un seuil donné."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    report = classification_report(
        y_test, y_pred, target_names=["normal", "suspect"],
        zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)

    return {
        "y_proba": y_proba,
        "y_pred": y_pred,
        "report": report,
        "confusion_matrix": cm,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
    }
