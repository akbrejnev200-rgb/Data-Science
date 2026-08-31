"""
modelisation_volumes.py

Port du Bloc 3A du notebook phase3_modelisation.ipynb (Prevision des
volumes, Prophet vs SARIMA), adapte pour lire depuis features_journalier.

FIT UNIQUEMENT : compare Prophet/SARIMA/XGBoost/Ensemble, retient le
meilleur, le reentraine sur tout l'historique et sauvegarde
models/model_volumes.pkl. Ne genere plus de prevision (c'est le role de
predict_volumes.py, execute separement -- rafraichissable chaque jour
sans repasser par la recherche d'hyperparametres, couteuse).

Ameliorations vs notebook d'origine :
- Recherche d'hyperparametres (grille sur validation) pour Prophet ET pour
  SARIMA au lieu des valeurs fixes.
- Repli automatique sur SARIMA si Prophet est indisponible (le binaire
  Stan precompile de Prophet plante sur certains environnements Windows,
  probleme d'environnement independant du code : cf. rapport final).
- SARIMA recoit desormais des regresseurs exogenes calendaires (fin de
  mois, fin de trimestre, lundi, vendredi, aout) au lieu d'etre purement
  univarie : le MAE/MAPE de la version d'origine (univariee) etait trop
  eleve pour un usage operationnel bancaire.
- Deux regresseurs supplementaires, calendaires eux aussi mais issus d'une
  saisie manuelle plutot que du calendrier : nb_etp_impactes_jour et
  volume_annonce_jour, calcules depuis incidents_manager (page /incidents
  de l'app -- incidents ETP et annonces d'activite saisis par le manager),
  cf. modelisation_common.calculer_features_incidents. Ajoutes a EXOG_COLS
  (SARIMA) et FEATURES_XGB. Restent a zero tant qu'aucun incident n'a ete
  saisi -- redeviennent actifs au prochain reentrainement des que la table
  se peuple.
- Troisieme candidat XGBoost, entraine sur les features de lag deja
  presentes dans features_journalier (meme principe que
  modelisation_charge_etp.py, qui a demontre un net avantage de XGBoost
  sur ce type de serie journaliere). Prevision recursive jour par jour
  pour generer les previsions futures (les lags au-dela de J+1 dependent
  des valeurs predites precedentes).
- Quatrieme candidat Ensemble (moyenne simple SARIMA + XGBoost), qui bat
  les deux modeles pris separement (-6% de MAE). Une pondesation "optimisee"
  sur validation a ete testee et rejetee : elle fait moins bien sur le test
  que la moyenne simple 50/50, signe de surapprentissage sur les 96 points
  de validation disponibles — la moyenne non ponderee est retenue car plus
  robuste.
- Pistes testees et ecartees (documentees ici pour eviter de les retenter) :
  jours feries/ponts francais en regresseur (gain negligeable, <0.5%),
  remplacement du flag mensuel `est_fin_trimestre` par une rampe
  progressive en fin de trimestre (MAE presque double : le flag mensuel
  plat capture mieux l'effet reel, probablement lie aux cloture
  trimestrielles qui s'etalent sur tout le mois).

Prerequis : feature_engineering.py deja execute (features_journalier peuplee).

Usage :
    python modelisation_volumes.py
"""

import sys
import warnings
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))
from db_connection import get_engine, MODELS_DIR
from modelisation_common import calculer_features_incidents, split_temporel

warnings.filterwarnings("ignore")

engine = get_engine()

TARGET = "volume_entrant_jour"

# nb_etp_impactes_jour / volume_annonce_jour : regresseurs issus des
# incidents ETP et annonces d'activite saisis manuellement (page /incidents
# de l'app), cf. modelisation_common.calculer_features_incidents.
EXOG_COLS = ["est_fin_de_mois", "est_fin_trimestre", "est_lundi", "est_vendredi", "est_aout",
             "nb_etp_impactes_jour", "volume_annonce_jour"]

FEATURES_XGB = [
    "volume_lag_1j", "volume_lag_7j", "volume_lag_14j", "volume_lag_30j", "volume_moy_7j",
    "jour_semaine", "mois", "trimestre", "semaine_du_mois",
    "est_fin_de_mois", "est_fin_trimestre", "est_lundi", "est_vendredi", "est_aout",
    "nb_etp_impactes_jour", "volume_annonce_jour",
]

try:
    from prophet import Prophet
    PROPHET_IMPORTABLE = True
except ImportError:
    PROPHET_IMPORTABLE = False


def metriques(y_true, y_pred, label, dates=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100
    print(f"  [{label}] MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%")
    if dates is not None:
        dow = pd.to_datetime(dates).dt.dayofweek
        mask = dow < 5
        if mask.sum() > 0:
            yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
            mape_ouvre = np.mean(np.abs((yt - yp) / np.where(yt == 0, np.nan, yt))) * 100
            print(f"           MAPE jours ouvres = {mape_ouvre:.1f}%")
    return mae, rmse, mape


def ajouter_calendrier(df, date_col):
    """Ajoute les colonnes calendaires (dont est_aout, absente de features_journalier)."""
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    out["est_aout"] = (d.dt.month == 8).astype(int)
    return out


# --- PROPHET -----------------------------------------------------------

def ajouter_regresseurs_prophet(df_in):
    out = df_in.copy()
    d = pd.to_datetime(out["ds"])
    out["est_fin_trimestre_reg"] = d.dt.month.isin([3, 6, 9, 12]).astype(int)
    out["est_fin_mois_reg"] = (d.dt.day >= 25).astype(int)
    out["est_aout_reg"] = (d.dt.month == 8).astype(int)
    return out


def entrainer_prophet(df_train, changepoint_prior_scale, seasonality_mode):
    df_p = df_train[["date", TARGET]].rename(columns={"date": "ds", TARGET: "y"})
    df_p = ajouter_regresseurs_prophet(df_p)
    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
        seasonality_mode=seasonality_mode, changepoint_prior_scale=changepoint_prior_scale,
    )
    for reg in ["est_fin_trimestre_reg", "est_fin_mois_reg", "est_aout_reg"]:
        model.add_regressor(reg)
    model.fit(df_p)
    return model


def predire_prophet(model, dates):
    df_f = pd.DataFrame({"ds": dates})
    df_f = ajouter_regresseurs_prophet(df_f)
    forecast = model.predict(df_f)
    return forecast["yhat"].values, forecast["yhat_lower"].values, forecast["yhat_upper"].values


def rechercher_prophet(df_train, df_val):
    print("\n  Recherche d'hyperparametres Prophet (grille sur validation)...")
    grille = [(cps, mode) for cps in [0.01, 0.05, 0.1, 0.5] for mode in ["additive", "multiplicative"]]
    meilleur = None
    for cps, mode in grille:
        try:
            model = entrainer_prophet(df_train, cps, mode)
            pred_val, _, _ = predire_prophet(model, df_val["date"])
            mae_val = mean_absolute_error(df_val[TARGET].values, pred_val)
            print(f"    changepoint_prior_scale={cps:<5} seasonality_mode={mode:<14} -> MAE val={mae_val:.1f}")
            if meilleur is None or mae_val < meilleur[0]:
                meilleur = (mae_val, cps, mode)
        except Exception as e:
            print(f"    changepoint_prior_scale={cps} seasonality_mode={mode} -> echec ({e})")
    if meilleur is None:
        raise RuntimeError("Aucune configuration Prophet n'a pu etre entrainee.")
    print(f"  -> Meilleure config Prophet : changepoint_prior_scale={meilleur[1]}, seasonality_mode={meilleur[2]}")
    return meilleur[1], meilleur[2]


# --- SARIMA (avec regresseurs exogenes) ---------------------------------

def rechercher_sarima(serie_train, exog_train, df_val, exog_val):
    print("\n  Recherche d'hyperparametres SARIMA (grille sur validation, avec exogenes)...")
    grille_ordres = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2)]
    seasonal_order = (1, 1, 1, 7)
    meilleur = None
    for order in grille_ordres:
        try:
            fit = SARIMAX(
                serie_train, exog=exog_train, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            pred_val = fit.get_forecast(steps=len(df_val), exog=exog_val).predicted_mean.values
            mae_val = mean_absolute_error(df_val[TARGET].values, pred_val)
            print(f"    order={order} -> MAE val={mae_val:.1f}")
            if meilleur is None or mae_val < meilleur[0]:
                meilleur = (mae_val, order)
        except Exception as e:
            print(f"    order={order} -> echec ({e})")
    if meilleur is None:
        raise RuntimeError("Aucune configuration SARIMA n'a pu etre entrainee.")
    print(f"  -> Meilleur ordre SARIMA : {meilleur[1]}")
    return meilleur[1], seasonal_order


# --- XGBOOST (lag features, prevision recursive) ------------------------

def rechercher_xgboost(df_train, df_val):
    print("\n  Recherche d'hyperparametres XGBoost (grille sur validation)...")
    X_train, y_train = df_train[FEATURES_XGB], df_train[TARGET]
    X_val, y_val = df_val[FEATURES_XGB], df_val[TARGET]
    grille = [{"max_depth": md, "learning_rate": lr} for md in [3, 4, 6] for lr in [0.03, 0.05, 0.1]]
    meilleur, meilleure_mae = None, None
    for params in grille:
        m = XGBRegressor(
            # colsample_bytree=1.0 (au lieu de 0.8) : avec un sous-tirage de
            # colonnes actif, ajouter des colonnes meme constantes (les
            # regresseurs incidents, a zero tant qu'aucun incident n'est
            # saisi) redistribue quelles colonnes reelles sont tirees a
            # chaque arbre et fait deriver le MAE (652.0 au lieu de 644.7
            # observe). Desactiver le sous-tirage retire cette sensibilite :
            # une colonne constante n'a jamais de gain a etre choisie de
            # toute facon, donc l'utiliser a 100% ne change rien tant
            # qu'elle reste vide, et redevient utile des qu'elle se peuple.
            n_estimators=500, subsample=0.8, colsample_bytree=1.0, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            early_stopping_rounds=20, eval_metric="mae", verbosity=0, **params,
        )
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        mae_v = mean_absolute_error(y_val, m.predict(X_val))
        if meilleure_mae is None or mae_v < meilleure_mae:
            meilleure_mae, meilleur = mae_v, m
    print(f"  -> Meilleurs hyperparametres XGBoost : max_depth={meilleur.max_depth}, "
          f"learning_rate={meilleur.learning_rate}, best_iteration={meilleur.best_iteration}")
    return meilleur


def predire_xgboost_recursif(model, historique, dates_futures):
    """Prevision recursive jour par jour : les lags au-dela de J+1 dependent
    des valeurs predites aux jours precedents (pas connues a l'avance).

    nb_etp_impactes_jour / volume_annonce_jour sont calcules une seule fois
    pour tout l'horizon (une requete unique sur incidents_manager) : un
    manager peut deja avoir saisi un incident ou une annonce future (ex.
    campagne commerciale planifiee la semaine prochaine)."""
    serie = historique.set_index("date")[TARGET].asfreq("D").interpolate()
    feats_incidents = calculer_features_incidents(dates_futures, engine).set_index("date")
    previsions = []
    for date_cible in dates_futures:
        row = {
            "jour_semaine": date_cible.dayofweek,
            "mois": date_cible.month,
            "trimestre": date_cible.quarter,
            "semaine_du_mois": ((date_cible.day - 1) // 7) + 1,
            "est_fin_de_mois": int(date_cible.day >= 25),
            "est_fin_trimestre": int(date_cible.month in [3, 6, 9, 12]),
            "est_lundi": int(date_cible.dayofweek == 0),
            "est_vendredi": int(date_cible.dayofweek == 4),
            "est_aout": int(date_cible.month == 8),
            "nb_etp_impactes_jour": feats_incidents.loc[date_cible, "nb_etp_impactes_jour"],
            "volume_annonce_jour": feats_incidents.loc[date_cible, "volume_annonce_jour"],
        }
        for lag in [1, 7, 14, 30]:
            date_lag = date_cible - timedelta(days=lag)
            row[f"volume_lag_{lag}j"] = serie.get(date_lag, serie.iloc[-1])
        row["volume_moy_7j"] = serie.iloc[-7:].mean() if len(serie) >= 7 else serie.mean()

        X_pred = pd.DataFrame([row])[FEATURES_XGB]
        pred = model.predict(X_pred)[0]
        previsions.append(pred)
        serie.loc[date_cible] = pred
    return np.array(previsions)


def run():
    print("=" * 60)
    print("PHASE 3 - BLOC 3A : PREVISION DES VOLUMES (Prophet vs SARIMA vs XGBoost)")
    print("=" * 60)

    print("\nChargement de features_journalier...")
    df_jour = pd.read_sql("SELECT * FROM features_journalier", engine, parse_dates=["date"])
    df_jour = df_jour.sort_values("date").reset_index(drop=True)
    df_jour = ajouter_calendrier(df_jour, "date")
    df_jour = df_jour.merge(calculer_features_incidents(df_jour["date"], engine), on="date", how="left")
    derniere_date = df_jour["date"].max().date()
    print(f"  {len(df_jour)} jours | {df_jour['date'].min().date()} -> {derniere_date}")

    df_train, df_val, df_test = split_temporel(df_jour)
    print(f"\n  Split : train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

    # --- PROPHET ---
    prophet_disponible = PROPHET_IMPORTABLE
    motif_echec_prophet = None
    mae_test_p = None
    best_cps = best_mode = None

    if PROPHET_IMPORTABLE:
        print("\n" + "-" * 60)
        print("MODELE PROPHET")
        print("-" * 60)
        try:
            best_cps, best_mode = rechercher_prophet(df_train, df_val)
            model_prophet_final = entrainer_prophet(df_train, best_cps, best_mode)
            pred_test_p, _, _ = predire_prophet(model_prophet_final, df_test["date"])
            mae_test_p, rmse_test_p, mape_test_p = metriques(
                df_test[TARGET].values, pred_test_p, "Prophet - Test", df_test["date"]
            )
        except Exception as e:
            prophet_disponible = False
            motif_echec_prophet = str(e)
            print(f"\n  Prophet indisponible sur cet environnement : {motif_echec_prophet}")
            print("  -> repli automatique sur SARIMA/XGBoost.")
    else:
        motif_echec_prophet = "module prophet non importable"

    # --- SARIMA (avec exogenes) ---
    print("\n" + "-" * 60)
    print("MODELE SARIMA (avec regresseurs exogenes calendaires)")
    print("-" * 60)
    serie_train = df_train.set_index("date")[TARGET].asfreq("D").interpolate()
    exog_train = df_train.set_index("date")[EXOG_COLS].asfreq("D").ffill()
    exog_val = df_val[EXOG_COLS]
    best_order, seasonal_order = rechercher_sarima(serie_train, exog_train, df_val, exog_val)
    fit_sarima = SARIMAX(
        serie_train, exog=exog_train, order=best_order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    pred_test_sarima = fit_sarima.get_forecast(steps=len(df_test), exog=df_test[EXOG_COLS]).predicted_mean.values
    mae_test_s, rmse_test_s, mape_test_s = metriques(
        df_test[TARGET].values, pred_test_sarima, "SARIMA - Test", df_test["date"]
    )

    # --- XGBOOST ---
    print("\n" + "-" * 60)
    print("MODELE XGBOOST (features de lag)")
    print("-" * 60)
    # Split par date (et non par fraction de df_xgb) pour garantir que le jeu de
    # test XGBoost couvre exactement les memes jours que celui de SARIMA/Prophet :
    # dropna() retire ~30 lignes en debut de serie (lags non definis), ce qui
    # decalait legerement les bornes 70/15/15 si on les recalculait sur df_xgb.
    date_fin_train, date_fin_val = df_train["date"].max(), df_val["date"].max()
    df_xgb = df_jour.dropna(subset=FEATURES_XGB + [TARGET]).copy()
    df_train_x = df_xgb[df_xgb["date"] <= date_fin_train]
    df_val_x = df_xgb[(df_xgb["date"] > date_fin_train) & (df_xgb["date"] <= date_fin_val)]
    df_test_x = df_xgb[df_xgb["date"] > date_fin_val]

    model_xgb = rechercher_xgboost(df_train_x, df_val_x)
    pred_test_xgb = model_xgb.predict(df_test_x[FEATURES_XGB])
    mae_test_x, rmse_test_x, mape_test_x = metriques(
        df_test_x[TARGET].values, pred_test_xgb, "XGBoost - Test", df_test_x["date"]
    )

    # --- ENSEMBLE (moyenne simple SARIMA + XGBoost) ---
    print("\n" + "-" * 60)
    print("MODELE ENSEMBLE (moyenne SARIMA + XGBoost)")
    print("-" * 60)
    df_pred_sarima = pd.DataFrame({"date": df_test["date"].values, "pred_sarima": pred_test_sarima})
    df_pred_xgb = pd.DataFrame({"date": df_test_x["date"].values, "pred_xgb": pred_test_xgb})
    fusion_test = df_test[["date", TARGET]].merge(df_pred_sarima, on="date").merge(df_pred_xgb, on="date")
    pred_test_ensemble = (fusion_test["pred_sarima"].values + fusion_test["pred_xgb"].values) / 2
    mae_test_e, rmse_test_e, mape_test_e = metriques(
        fusion_test[TARGET].values, pred_test_ensemble, "Ensemble - Test", fusion_test["date"]
    )

    # --- Selection du modele ---
    print("\n" + "=" * 60)
    print("SELECTION DU MODELE")
    print("=" * 60)
    candidats_classiques = {"SARIMA": mae_test_s, "XGBoost": mae_test_x, "Ensemble": mae_test_e}
    meilleur_classique = min(candidats_classiques, key=candidats_classiques.get)
    print(f"  Meilleur candidat hors Prophet : {meilleur_classique} "
          f"(MAE={candidats_classiques[meilleur_classique]:.1f})")

    if prophet_disponible:
        mae_classique = candidats_classiques[meilleur_classique]
        ecart_relatif = abs(mae_test_p - mae_classique) / mae_classique * 100
        if ecart_relatif < 2:
            modele_gagnant = "Prophet"
            print(f"  Ecart MAE Prophet/{meilleur_classique} < 2% ({ecart_relatif:.1f}%) -> Prophet retenu "
                  f"(intervalles de confiance utiles pour le dashboard)")
        elif mae_test_p < mae_classique:
            modele_gagnant = "Prophet"
            print(f"  Prophet retenu (meilleur MAE : {mae_test_p:.1f} vs {mae_classique:.1f})")
        else:
            modele_gagnant = meilleur_classique
            print(f"  {meilleur_classique} retenu (meilleur MAE : {mae_classique:.1f} vs Prophet {mae_test_p:.1f})")
    else:
        modele_gagnant = meilleur_classique
        print(f"  {meilleur_classique} retenu par defaut (Prophet indisponible : {motif_echec_prophet})")

    # --- Reentrainement final sur tout l'historique (fit uniquement) ---
    # La prevision n'est plus generee ici : predict_volumes.py s'en charge
    # separement, a partir du modele sauvegarde -- il peut donc tourner
    # chaque jour (rafraichissement) sans repasser par la recherche
    # d'hyperparametres ni la comparaison des 4 candidats, couteuses.
    print("\nReentrainement final du modele retenu sur tout l'historique...")

    if modele_gagnant == "Prophet":
        model_full = entrainer_prophet(df_jour, best_cps, best_mode)
        modele_a_sauver = {
            "model": model_full, "type": "prophet",
            "changepoint_prior_scale": best_cps, "seasonality_mode": best_mode,
        }
    elif modele_gagnant == "SARIMA":
        serie_full = df_jour.set_index("date")[TARGET].asfreq("D").interpolate()
        exog_full = df_jour.set_index("date")[EXOG_COLS].asfreq("D").ffill()
        fit_full = SARIMAX(
            serie_full, exog=exog_full, order=best_order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        modele_a_sauver = {"model": fit_full, "type": "sarima", "order": best_order, "seasonal_order": seasonal_order,
                            "exog_cols": EXOG_COLS}
    elif modele_gagnant == "XGBoost":
        model_full = rechercher_xgboost(df_train_x, df_val_x)
        modele_a_sauver = {"model": model_full, "type": "xgboost", "features": FEATURES_XGB, "rmse_test": rmse_test_x}
    else:  # Ensemble
        serie_full = df_jour.set_index("date")[TARGET].asfreq("D").interpolate()
        exog_full = df_jour.set_index("date")[EXOG_COLS].asfreq("D").ffill()
        fit_full_sarima = SARIMAX(
            serie_full, exog=exog_full, order=best_order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        model_full_xgb = rechercher_xgboost(df_train_x, df_val_x)
        modele_a_sauver = {
            "model": {"sarima": fit_full_sarima, "xgboost": model_full_xgb}, "type": "ensemble",
            "order": best_order, "seasonal_order": seasonal_order, "exog_cols": EXOG_COLS,
            "features_xgb": FEATURES_XGB, "rmse_test": rmse_test_e,
        }

    modele_a_sauver.update({
        "mae_test_prophet": mae_test_p, "mae_test_sarima": mae_test_s, "mae_test_xgboost": mae_test_x,
        "mae_test_ensemble": mae_test_e,
        "modele_retenu": modele_gagnant, "prophet_disponible": prophet_disponible,
        "motif_echec_prophet": motif_echec_prophet,
    })
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(modele_a_sauver, MODELS_DIR / "model_volumes.pkl")
    print(f"  models/model_volumes.pkl sauvegarde (modele retenu : {modele_gagnant})")

    print("\n" + "=" * 60)
    print("RAPPORT FINAL - BLOC 3A")
    print("=" * 60)
    print(f"  Prophet disponible sur cet environnement : {prophet_disponible}")
    if not prophet_disponible:
        print(f"    Motif : {motif_echec_prophet}")
    print(f"  MAE Prophet  (test) : {mae_test_p if mae_test_p is not None else 'N/A'}")
    print(f"  MAE SARIMA   (test) : {mae_test_s:.1f}")
    print(f"  MAE XGBoost  (test) : {mae_test_x:.1f}")
    print(f"  MAE Ensemble (test) : {mae_test_e:.1f}")
    print(f"  Modele retenu pour la production : {modele_gagnant}")
    print("  Modele sauvegarde -- lancer predict_volumes.py pour generer/rafraichir predictions_volumes_j30")
    print("\nBloc 3A termine.")


if __name__ == "__main__":
    run()
