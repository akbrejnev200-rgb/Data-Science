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


def calibrate_threshold(model, X_val, y_val, target_recall=0.80):
    """Choisit le seuil métier sur la validation : le seuil le plus élevé (donc
    la meilleure précision) qui atteint au moins `target_recall` sur la classe
    suspecte.

    En détection AML, la contrainte vient de la conformité — un niveau de rappel
    minimal à garantir — et la précision est ce qu'on obtient en conséquence.
    Aucune valeur de F-beta ne fixe le rappel ; un rappel cible, si. Si aucun
    seuil n'atteint la cible, on retient le seuil le plus bas (rappel maximal).

    Note : la validation faisant partie de l'entraînement du modèle final, le
    rappel réel sur le test est sensiblement plus bas que `target_recall`.
    """
    val_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)

    # precision_recall_curve : recalls décroît quand le seuil croît ; thresholds
    # est aligné sur precisions[:-1] / recalls[:-1] (dernier point : pas de seuil).
    atteint_cible = recalls[:-1] >= target_recall
    best_idx = int(np.max(np.flatnonzero(atteint_cible))) if atteint_cible.any() else 0
    best_threshold = thresholds[best_idx]

    threshold_table = {
        cible: {
            "threshold": thresholds[np.argmin(np.abs(recalls[:-1] - cible))],
            "precision": precisions[np.argmin(np.abs(recalls[:-1] - cible))],
            "recall": recalls[np.argmin(np.abs(recalls[:-1] - cible))],
        }
        for cible in [0.5, 0.6, 0.7, 0.8, 0.9]
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
