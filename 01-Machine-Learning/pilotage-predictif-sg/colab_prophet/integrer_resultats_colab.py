"""
integrer_resultats_colab.py

Compare le modele Bloc 3A entraine localement (SARIMA/XGBoost, Prophet
indisponible) avec celui rapporte de Google Colab (Prophet/SARIMA/XGBoost,
Prophet disponible la-bas), et ne remplace le modele local QUE si le
resultat Colab est reellement meilleur (MAE test plus bas).

Prerequis : avoir place dans `colab_prophet/resultats/` les 3 fichiers
telecharges depuis le notebook Colab :
    - model_volumes.pkl
    - predictions_volumes_j30.csv
    - metriques_bloc3a.txt (informatif, non relu par ce script)

Usage :
    python integrer_resultats_colab.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))
from db_connection import get_engine, MODELS_DIR

RESULTATS_DIR = Path(__file__).resolve().parent / "resultats"

CLE_MAE = {"Prophet": "mae_test_prophet", "SARIMA": "mae_test_sarima", "XGBoost": "mae_test_xgboost",
           "Ensemble": "mae_test_ensemble"}


def mae_du_modele_retenu(payload):
    return payload[CLE_MAE[payload["modele_retenu"]]]


def run():
    print("=" * 60)
    print("INTEGRATION DES RESULTATS COLAB (Bloc 3A)")
    print("=" * 60)

    chemin_local = MODELS_DIR / "model_volumes.pkl"
    chemin_colab = RESULTATS_DIR / "model_volumes.pkl"
    csv_colab = RESULTATS_DIR / "predictions_volumes_j30.csv"

    if not chemin_colab.exists() or not csv_colab.exists():
        print(f"\nFichiers Colab introuvables dans {RESULTATS_DIR}")
        print("Attendu : model_volumes.pkl et predictions_volumes_j30.csv")
        return

    if not chemin_local.exists():
        print(f"\nAucun modele local trouve dans {chemin_local} : le modele Colab est adopte directement.")
        adopter_colab = True
        payload_local = None
    else:
        payload_local = joblib.load(chemin_local)
        payload_colab = joblib.load(chemin_colab)
        mae_local = mae_du_modele_retenu(payload_local)
        mae_colab = mae_du_modele_retenu(payload_colab)

        print(f"\nModele local  : {payload_local['modele_retenu']:<10} MAE test = {mae_local:.1f}")
        print(f"Modele Colab  : {payload_colab['modele_retenu']:<10} MAE test = {mae_colab:.1f}"
              f"  (Prophet disponible : {payload_colab.get('prophet_disponible')})")

        adopter_colab = mae_colab < mae_local

    if adopter_colab:
        payload_colab = joblib.load(chemin_colab)
        joblib.dump(payload_colab, chemin_local)
        print(f"\n-> Modele Colab retenu ({payload_colab['modele_retenu']}), "
              f"models/model_volumes.pkl remplace.")

        df_prevision = pd.read_csv(csv_colab, parse_dates=["date"])
        engine = get_engine()
        df_prevision.to_sql("predictions_volumes_j30", engine, if_exists="replace", index=False)
        print(f"-> predictions_volumes_j30 mise a jour ({len(df_prevision)} lignes) avec les previsions Colab.")
    else:
        print(f"\n-> Modele local conserve ({payload_local['modele_retenu']}, "
              f"MAE={mae_du_modele_retenu(payload_local):.1f} < Colab). Aucun changement.")

    print("\nTermine.")


if __name__ == "__main__":
    run()
