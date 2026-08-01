"""
Entraînement des modèles : Isolation Forest (non supervisé) et Random Forest (supervisé).
"""

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score


def train_isolation_forest(X_scaled, contamination_rate, random_state=42):
    """Entraîne un Isolation Forest sur les features standardisées."""
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination_rate,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return model


def select_best_depth(X_train, y_train, X_val, y_val, depths=(8, 12), random_state=42):
    """Compare plusieurs profondeurs d'arbre sur le jeu de validation, retourne la meilleure."""
    best_auc_pr = -1
    best_depth = None
    results = {}

    for depth in depths:
        candidate = RandomForestClassifier(
            n_estimators=300, max_depth=depth,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
        candidate.fit(X_train, y_train)
        val_proba = candidate.predict_proba(X_val)[:, 1]
        val_auc_pr = average_precision_score(y_val, val_proba)
        results[depth] = val_auc_pr

        if val_auc_pr > best_auc_pr:
            best_auc_pr = val_auc_pr
            best_depth = depth

    return best_depth, results


def train_random_forest(X_train, y_train, max_depth, random_state=42):
    """Entraîne un Random Forest avec les hyperparamètres retenus."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
