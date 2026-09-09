"""
Entraînement des modèles : Isolation Forest (non supervisé), Random Forest et XGBoost (supervisés).
"""

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier


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
    """Compare plusieurs profondeurs d'arbre (Random Forest) sur le jeu de validation."""
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


def select_best_xgboost_params(X_train, y_train, X_val, y_val, random_state=42):
    """Compare plusieurs configurations XGBoost sur le jeu de validation."""
    # scale_pos_weight = ratio negatif/positif, équivalent de class_weight="balanced" pour XGBoost
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    candidats = [
        {"max_depth": 4, "learning_rate": 0.1},
        {"max_depth": 6, "learning_rate": 0.1},
        {"max_depth": 6, "learning_rate": 0.05},
    ]

    best_auc_pr = -1
    best_params = None
    results = {}

    for params in candidats:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        val_auc_pr = average_precision_score(y_val, val_proba)
        key = f"depth={params['max_depth']}, lr={params['learning_rate']}"
        results[key] = val_auc_pr

        if val_auc_pr > best_auc_pr:
            best_auc_pr = val_auc_pr
            best_params = params

    return best_params, results, scale_pos_weight


def train_xgboost(X_train, y_train, params, scale_pos_weight, random_state=42):
    """Entraîne un XGBoost avec les hyperparamètres retenus."""
    model = XGBClassifier(
        n_estimators=300,
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
