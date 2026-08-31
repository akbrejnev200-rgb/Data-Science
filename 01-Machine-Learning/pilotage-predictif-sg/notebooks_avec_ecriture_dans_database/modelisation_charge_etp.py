"""
modelisation_charge_etp.py

Port du Bloc 3B du notebook phase3_modelisation.ipynb (Prevision de la
charge par ETP, Ridge vs ElasticNet vs XGBoost), adapte pour lire depuis
features_journalier.

FIT UNIQUEMENT : compare Ridge/ElasticNet/XGBoost, retient le meilleur,
sauvegarde models/model_charge_etp.pkl. Ne genere plus le backtest (c'est
le role de predict_charge.py, execute separement).

Amelioration vs notebook d'origine : recherche d'hyperparametres par
TimeSeriesSplit (au lieu des valeurs fixes alpha=1.0 / n_estimators=300
etc. choisies a la main), et features cycliques (sin/cos) supplementaires
pour mieux capter la saisonnalite hebdomadaire/mensuelle. Deux regresseurs
supplementaires issus d'une saisie manuelle (page /incidents de l'app) :
nb_etp_impactes_jour et volume_annonce_jour, calcules depuis
incidents_manager (cf. modelisation_common.calculer_features_incidents).

Prerequis : feature_engineering.py deja execute (features_journalier peuplee).

Usage :
    python modelisation_charge_etp.py
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))
from db_connection import get_engine, MODELS_DIR
from modelisation_common import calculer_features_incidents, split_temporel

warnings.filterwarnings("ignore")

engine = get_engine()

TARGET = "charge_par_etp"

# nb_etp_impactes_jour / volume_annonce_jour : regresseurs issus des
# incidents ETP et annonces d'activite saisis manuellement (page /incidents
# de l'app), cf. modelisation_common.calculer_features_incidents.
FEATURES_BASE = [
    "volume_lag_1j", "volume_lag_7j", "volume_lag_14j", "volume_lag_30j",
    "etp_lag_1j", "etp_lag_7j", "etp_lag_14j", "etp_lag_30j",
    "volume_moy_7j", "charge_moy_7j",
    "jour_semaine", "mois", "trimestre", "semaine_du_mois",
    "est_fin_de_mois", "est_fin_trimestre", "est_lundi", "est_vendredi",
    "etp_disponible", "taux_complexes",
    "nb_etp_impactes_jour", "volume_annonce_jour",
]
FEATURES_CYCLIQUES = ["jour_semaine_sin", "jour_semaine_cos", "mois_sin", "mois_cos"]
FEATURES = FEATURES_BASE + FEATURES_CYCLIQUES


def ajouter_features_cycliques(df):
    df = df.copy()
    df["jour_semaine_sin"] = np.sin(2 * np.pi * df["jour_semaine"] / 7)
    df["jour_semaine_cos"] = np.cos(2 * np.pi * df["jour_semaine"] / 7)
    df["mois_sin"] = np.sin(2 * np.pi * df["mois"] / 12)
    df["mois_cos"] = np.cos(2 * np.pi * df["mois"] / 12)
    return df


def metriques(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  [{label}] MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")
    return mae, rmse, mse, r2


def walk_forward(X, y, model_class, model_kwargs, taille_fenetre=120, taille_pred=14):
    maes = []
    i = taille_fenetre
    while i + taille_pred <= len(X):
        X_tr, y_tr = X.iloc[i - taille_fenetre:i], y.iloc[i - taille_fenetre:i]
        X_pr, y_pr = X.iloc[i:i + taille_pred], y.iloc[i:i + taille_pred]
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_pr_sc = sc.transform(X_pr)
        m = model_class(**model_kwargs)
        m.fit(X_tr_sc, y_tr)
        maes.append(mean_absolute_error(y_pr, m.predict(X_pr_sc)))
        i += taille_pred
    return maes


def run():
    print("=" * 60)
    print("PHASE 3 - BLOC 3B : PREVISION DE LA CHARGE PAR ETP")
    print("=" * 60)

    print("\nChargement de features_journalier...")
    df_jour = pd.read_sql("SELECT * FROM features_journalier", engine, parse_dates=["date"])
    df_jour = df_jour.sort_values("date").reset_index(drop=True)
    df_jour = ajouter_features_cycliques(df_jour)
    df_jour = df_jour.merge(calculer_features_incidents(df_jour["date"], engine), on="date", how="left")

    df_model = df_jour[FEATURES + [TARGET, "date"]].dropna().copy()
    print(f"  Lignes utilisables apres dropna : {len(df_model)} / {len(df_jour)}")

    df_train, df_val, df_test = split_temporel(df_model)
    print(f"  Split : train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

    X_train, y_train = df_train[FEATURES], df_train[TARGET]
    X_val, y_val = df_val[FEATURES], df_val[TARGET]
    X_test, y_test = df_test[FEATURES], df_test[TARGET]

    tscv = TimeSeriesSplit(n_splits=5)

    # --- MODELE 1 : RIDGE (grille d'hyperparametres) ---
    print("\n" + "-" * 60)
    print("MODELE 1 - RIDGE (GridSearchCV, TimeSeriesSplit)")
    print("-" * 60)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    grid_ridge = GridSearchCV(
        Ridge(random_state=42), {"alpha": [0.1, 1.0, 5.0, 10.0, 50.0]},
        cv=tscv, scoring="neg_mean_absolute_error",
    )
    grid_ridge.fit(X_train_sc, y_train)
    model_ridge = grid_ridge.best_estimator_
    print(f"  Meilleur alpha Ridge : {grid_ridge.best_params_['alpha']}")

    pred_val_ridge = model_ridge.predict(X_val_sc)
    pred_test_ridge = model_ridge.predict(X_test_sc)
    metriques(y_val, pred_val_ridge, "Ridge - Validation")
    mae_ridge, rmse_ridge, mse_ridge, r2_ridge = metriques(y_test, pred_test_ridge, "Ridge - Test")

    # --- MODELE 2 : ELASTIC NET (grille d'hyperparametres) ---
    print("\n" + "-" * 60)
    print("MODELE 2 - ELASTIC NET (GridSearchCV, TimeSeriesSplit)")
    print("-" * 60)
    grid_elastic = GridSearchCV(
        ElasticNet(random_state=42, max_iter=5000),
        {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
        cv=tscv, scoring="neg_mean_absolute_error",
    )
    grid_elastic.fit(X_train_sc, y_train)
    model_elastic = grid_elastic.best_estimator_
    print(f"  Meilleurs hyperparametres Elastic Net : {grid_elastic.best_params_}")

    pred_val_elastic = model_elastic.predict(X_val_sc)
    pred_test_elastic = model_elastic.predict(X_test_sc)
    metriques(y_val, pred_val_elastic, "Elastic Net - Validation")
    mae_elastic, rmse_elastic, mse_elastic, r2_elastic = metriques(y_test, pred_test_elastic, "Elastic Net - Test")

    coeffs = pd.Series(model_elastic.coef_, index=FEATURES)
    elimines = coeffs[coeffs == 0].index.tolist()
    print(f"  Variables eliminees par Elastic Net (coef=0) : {elimines}")

    # --- MODELE 3 : XGBOOST (grille d'hyperparametres avec early stopping) ---
    print("\n" + "-" * 60)
    print("MODELE 3 - XGBOOST (recherche sur validation, early stopping)")
    print("-" * 60)
    grille_xgb = [
        {"max_depth": md, "learning_rate": lr}
        for md in [3, 4, 6]
        for lr in [0.03, 0.05, 0.1]
    ]
    meilleur_xgb, meilleure_mae_xgb = None, None
    for params in grille_xgb:
        m = XGBRegressor(
            # colsample_bytree=1.0 : meme raison que dans modelisation_volumes.py
            # -- neutralise la sensibilite aux regresseurs incidents encore a
            # zero (colonnes constantes redistribuant le tirage aleatoire des
            # autres colonnes a chaque arbre).
            n_estimators=500, subsample=0.8, colsample_bytree=1.0,
            min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, early_stopping_rounds=20, eval_metric="mae", verbosity=0,
            **params,
        )
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        mae_v = mean_absolute_error(y_val, m.predict(X_val))
        if meilleure_mae_xgb is None or mae_v < meilleure_mae_xgb:
            meilleure_mae_xgb, meilleur_xgb = mae_v, m
    print(f"  Meilleurs hyperparametres XGBoost : max_depth={meilleur_xgb.max_depth}, "
          f"learning_rate={meilleur_xgb.learning_rate}, best_iteration={meilleur_xgb.best_iteration}")
    model_xgb = meilleur_xgb

    pred_val_xgb = model_xgb.predict(X_val)
    pred_test_xgb = model_xgb.predict(X_test)
    metriques(y_val, pred_val_xgb, "XGBoost - Validation")
    mae_xgb, rmse_xgb, mse_xgb, r2_xgb = metriques(y_test, pred_test_xgb, "XGBoost - Test")

    # --- WALK-FORWARD VALIDATION (stabilite Ridge/ElasticNet) ---
    print("\n" + "-" * 60)
    print("WALK-FORWARD VALIDATION (train, fenetre 120j / pas 14j)")
    print("-" * 60)
    maes_ridge_wf = walk_forward(X_train, y_train, Ridge, {"alpha": grid_ridge.best_params_["alpha"], "random_state": 42})
    maes_elastic_wf = walk_forward(
        X_train, y_train, ElasticNet,
        {"alpha": grid_elastic.best_params_["alpha"], "l1_ratio": grid_elastic.best_params_["l1_ratio"],
         "random_state": 42, "max_iter": 5000},
    )
    print(f"  Ridge       : MAE walk-forward moyen = {np.mean(maes_ridge_wf):.2f} ({len(maes_ridge_wf)} fenetres)")
    print(f"  Elastic Net : MAE walk-forward moyen = {np.mean(maes_elastic_wf):.2f} ({len(maes_elastic_wf)} fenetres)")

    # --- COMPARAISON ET SELECTION ---
    print("\n" + "=" * 60)
    print("COMPARAISON RIDGE vs ELASTIC NET vs XGBOOST (test)")
    print("=" * 60)
    comparaison = pd.DataFrame({
        "Modele": ["Ridge", "Elastic Net", "XGBoost"],
        "MAE": [mae_ridge, mae_elastic, mae_xgb],
        "RMSE": [rmse_ridge, rmse_elastic, rmse_xgb],
        "R2": [r2_ridge, r2_elastic, r2_xgb],
    })
    print(comparaison.to_string(index=False))

    if mae_elastic <= mae_ridge:
        modele_lineaire_retenu, mae_lineaire_retenu = "Elastic Net", mae_elastic
    else:
        modele_lineaire_retenu, mae_lineaire_retenu = "Ridge", mae_ridge
    print(f"\n  Modele lineaire retenu (reference) : {modele_lineaire_retenu} (MAE={mae_lineaire_retenu:.2f})")

    if mae_xgb < mae_lineaire_retenu:
        gain_pct = (mae_lineaire_retenu - mae_xgb) / mae_lineaire_retenu * 100
        modele_final, model_final_obj = "XGBoost", model_xgb
        print(f"  XGBoost apporte un gain de {gain_pct:.1f}% -> retenu pour la production")
    else:
        modele_final = modele_lineaire_retenu
        model_final_obj = model_elastic if modele_lineaire_retenu == "Elastic Net" else model_ridge
        print(f"  {modele_lineaire_retenu} retenu pour la production (plus simple/interpretable)")

    # --- Export du modele (FIT uniquement) ---
    # Les predictions/backtest (predictions_charge_etp) sont generees
    # separement par predict_charge.py, qui peut tourner chaque jour sans
    # repasser par les recherches d'hyperparametres ci-dessus. On sauvegarde
    # les dates de coupure train/val/test pour que predict_charge.py puisse
    # etiqueter correctement les nouvelles lignes (tout ce qui est
    # posterieur a la coupure test d'origine = "test", jamais vu a
    # l'entrainement).
    if modele_final == "XGBoost":
        payload = {"model": model_xgb, "features": FEATURES, "scaler": None, "type": "xgboost"}
    elif modele_final == "Elastic Net":
        payload = {"model": model_elastic, "features": FEATURES, "scaler": scaler, "type": "elastic_net"}
    else:
        payload = {"model": model_ridge, "features": FEATURES, "scaler": scaler, "type": "ridge"}
    payload.update({
        "mae_test": {"ridge": mae_ridge, "elastic_net": mae_elastic, "xgboost": mae_xgb}[
            {"Ridge": "ridge", "Elastic Net": "elastic_net", "XGBoost": "xgboost"}[modele_final]
        ],
        "modele_retenu": modele_final,
        "date_fin_train": df_train["date"].max(),
        "date_fin_val": df_val["date"].max(),
    })
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(payload, MODELS_DIR / "model_charge_etp.pkl")
    print(f"  models/model_charge_etp.pkl sauvegarde (modele retenu : {modele_final})")

    print("\n" + "=" * 60)
    print("RAPPORT FINAL - BLOC 3B")
    print("=" * 60)
    print(f"  Modele retenu pour la production : {modele_final}")
    print(f"  Ridge       -> MAE={mae_ridge:.2f}  R2={r2_ridge:.3f}  (alpha={grid_ridge.best_params_['alpha']})")
    print(f"  Elastic Net -> MAE={mae_elastic:.2f}  R2={r2_elastic:.3f}  ({grid_elastic.best_params_})")
    print(f"  XGBoost     -> MAE={mae_xgb:.2f}  R2={r2_xgb:.3f}")
    print(f"  Walk-forward Ridge       : {np.mean(maes_ridge_wf):.2f}")
    print(f"  Walk-forward Elastic Net : {np.mean(maes_elastic_wf):.2f}")
    print("\nBloc 3B termine.")


if __name__ == "__main__":
    run()
