"""
predict_volumes.py

PREDICT UNIQUEMENT pour le Bloc 3A : charge le modele deja entraine
(models/model_volumes.pkl, produit par modelisation_volumes.py), genere
une prevision J+1 a J+30 et (re)ecrit predictions_volumes_j30.

Cle : l'horizon part du DERNIER JOUR REEL de features_journalier
(derniere_date), jamais de la date calendaire reelle -- c'est le point
qui manquait dans l'ancienne version fit+predict monolithique (bug
diagnostique : les previsions etaient ancrees sur la date du jour ou le
script tournait, provoquant un ecart de plusieurs semaines avec le dernier
jour reel des donnees).

Ce script est concu pour tourner chaque jour (apres l'ingestion de
nouvelles donnees), sans jamais reentrainer les modeles -- contrairement a
modelisation_volumes.py, il ne fait aucune recherche d'hyperparametres.

Prerequis : models/model_volumes.pkl deja genere (modelisation_volumes.py).

Usage :
    python predict_volumes.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))
from db_connection import get_engine, MODELS_DIR

from modelisation_common import calculer_features_incidents
from modelisation_volumes import TARGET, ajouter_calendrier, predire_prophet, predire_xgboost_recursif

import joblib

engine = get_engine()

NOM_JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
HORIZON_JOURS = 30


def _exog_futur(dates_futures):
    exog = pd.DataFrame({
        "est_fin_de_mois": [int(d.day >= 25) for d in dates_futures],
        "est_fin_trimestre": [int(d.month in [3, 6, 9, 12]) for d in dates_futures],
        "est_lundi": [int(d.dayofweek == 0) for d in dates_futures],
        "est_vendredi": [int(d.dayofweek == 4) for d in dates_futures],
        "est_aout": [int(d.month == 8) for d in dates_futures],
    }, index=dates_futures)
    # nb_etp_impactes_jour / volume_annonce_jour : un manager peut deja avoir
    # saisi un incident ou une annonce couvrant ces dates futures.
    feats_incidents = calculer_features_incidents(dates_futures, engine).set_index("date")
    exog["nb_etp_impactes_jour"] = feats_incidents.loc[dates_futures, "nb_etp_impactes_jour"].values
    exog["volume_annonce_jour"] = feats_incidents.loc[dates_futures, "volume_annonce_jour"].values
    return exog


def run():
    print("=" * 60)
    print("PREDICT - BLOC 3A : PREVISION DES VOLUMES (J+1 a J+30)")
    print("=" * 60)

    print("\nChargement du modele entraine (models/model_volumes.pkl)...")
    modele = joblib.load(MODELS_DIR / "model_volumes.pkl")
    cle_mae = f"mae_test_{modele['type']}"
    print(f"  Modele retenu : {modele['modele_retenu']} (MAE test={modele.get(cle_mae):.1f})")

    print("\nChargement de features_journalier (dernieres donnees connues)...")
    df_jour = pd.read_sql("SELECT * FROM features_journalier", engine, parse_dates=["date"])
    df_jour = df_jour.sort_values("date").reset_index(drop=True)
    df_jour = ajouter_calendrier(df_jour, "date")
    derniere_date = df_jour["date"].max()
    print(f"  Dernier jour reel : {derniere_date.date()}")

    dates_candidates = pd.date_range(start=derniere_date + pd.Timedelta(days=1), periods=45, freq="D")
    dates_futures = dates_candidates[dates_candidates.dayofweek < 5][:HORIZON_JOURS]
    print(f"  Prevision generee du {dates_futures[0].date()} au {dates_futures[-1].date()} "
          f"({len(dates_futures)} jours ouvres)")

    print(f"\nPrediction avec le modele {modele['type']}...")
    if modele["type"] == "prophet":
        yhat, yhat_lower, yhat_upper = predire_prophet(modele["model"], pd.Series(dates_futures))
    elif modele["type"] == "sarima":
        exog_futur = _exog_futur(dates_futures)
        forecast = modele["model"].get_forecast(steps=len(dates_futures), exog=exog_futur)
        yhat = forecast.predicted_mean.values
        ci = forecast.conf_int(alpha=0.2)
        yhat_lower, yhat_upper = ci.iloc[:, 0].values, ci.iloc[:, 1].values
    elif modele["type"] == "xgboost":
        yhat = predire_xgboost_recursif(modele["model"], df_jour[["date", TARGET]], dates_futures)
        marge = modele["rmse_test"]
        yhat_lower, yhat_upper = yhat - marge, yhat + marge
    else:  # ensemble
        exog_futur = _exog_futur(dates_futures)
        yhat_sarima = modele["model"]["sarima"].get_forecast(steps=len(dates_futures), exog=exog_futur).predicted_mean.values
        yhat_xgb = predire_xgboost_recursif(modele["model"]["xgboost"], df_jour[["date", TARGET]], dates_futures)
        yhat = (yhat_sarima + yhat_xgb) / 2
        marge = modele["rmse_test"]
        yhat_lower, yhat_upper = yhat - marge, yhat + marge

    df_prevision = pd.DataFrame({
        "date": dates_futures,
        "horizon_j": range(1, len(dates_futures) + 1),
        "jour_semaine": [NOM_JOURS[d.weekday()] for d in dates_futures],
        "volume_prevu": np.round(yhat).astype(int),
        "volume_prevu_min": np.round(yhat_lower).astype(int),
        "volume_prevu_max": np.round(yhat_upper).astype(int),
    })

    print("\nEcriture dans PostgreSQL (predictions_volumes_j30)...")
    df_prevision.to_sql("predictions_volumes_j30", engine, if_exists="replace", index=False)
    print(f"  predictions_volumes_j30 : {len(df_prevision)} lignes ecrites")
    print(f"  Volume moyen prevu : {df_prevision['volume_prevu'].mean():.0f} dossiers/jour")

    print("\nPredict Bloc 3A termine.")


if __name__ == "__main__":
    run()
