"""
predict_anomalies.py

PREDICT UNIQUEMENT pour le Bloc 3D : charge le modele deja entraine
(models/model_isolation_forest.pkl) et (re)ecrit anomalies_detectees.

Contrairement aux Blocs 3B/3C, pas de notion de train/val/test a etiqueter
ici : l'Isolation Forest est un modele non supervise, scorer l'ensemble du
dataset avec le modele deja ajuste est le comportement normal (aucune
fuite -- il n'y a pas d'etiquette cachee a "predire" depuis un jeu de
test).

Prerequis : models/model_isolation_forest.pkl deja genere
(modelisation_anomalies.py).

Usage :
    python predict_anomalies.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))
from db_connection import get_engine, MODELS_DIR

engine = get_engine()


def run():
    print("=" * 60)
    print("PREDICT - BLOC 3D : DETECTION D'ANOMALIES")
    print("=" * 60)

    print("\nChargement du modele entraine (models/model_isolation_forest.pkl)...")
    modele = joblib.load(MODELS_DIR / "model_isolation_forest.pkl")
    print(f"  n_estimators={modele['n_estimators']}  contamination={modele['contamination']}")

    print("\nChargement de features_detail...")
    df = pd.read_sql("SELECT * FROM features_detail", engine, parse_dates=["Date_Creation"])
    cols_meta = ["ID_Tache", "Matricule_Agent", "Service", "Date_Creation", "Type_Processus", "Complexite"]
    df_model = df[modele["features"] + cols_meta].dropna(subset=modele["features"]).copy()
    print(f"  Lignes utilisables : {len(df_model)} / {len(df)}")

    print("\nScoring de l'ensemble du dataset...")
    X = modele["scaler"].transform(df_model[modele["features"]])
    df_model["score_anomalie"] = modele["model"].decision_function(X)
    df_model["est_anomalie"] = (modele["model"].predict(X) == -1).astype(int)
    taux_global = df_model["est_anomalie"].mean()
    print(f"  Taux d'anomalies detectees (global) : {taux_global * 100:.2f}%")
    print("\n  Repartition par service :")
    print((df_model.groupby("Service")["est_anomalie"].mean().sort_values(ascending=False) * 100).to_string())

    print("\nEcriture dans PostgreSQL (anomalies_detectees)...")
    colonnes_export = ["ID_Tache", "Matricule_Agent", "Service", "Date_Creation", "Type_Processus",
                        "Complexite", "Temps_Passe_Declare_Min", "Volume_Dossiers",
                        "productivite_dossiers_par_heure", "score_anomalie", "est_anomalie"]
    df_model[colonnes_export].to_sql("anomalies_detectees", engine, if_exists="replace", index=False, chunksize=5000)
    print(f"  anomalies_detectees : {len(df_model)} lignes ecrites")

    print("\nPredict Bloc 3D termine.")


if __name__ == "__main__":
    run()
