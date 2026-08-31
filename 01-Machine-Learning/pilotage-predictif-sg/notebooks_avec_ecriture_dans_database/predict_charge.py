"""
predict_charge.py

PREDICT UNIQUEMENT pour le Bloc 3B : charge le modele deja entraine
(models/model_charge_etp.pkl) et (re)ecrit le backtest
predictions_charge_etp, avec la colonne "split" correctement etiquetee a
partir des dates de coupure figees a l'entrainement (train/validation/
test) -- toute ligne posterieure a la date de fin de validation d'origine
est "test", y compris les jours ajoutes depuis le dernier entrainement,
par definition jamais vus par le modele.

Note : ce backtest n'est pas ce qui alimente la carte "Charge globale/ETP"
de la page Surcharge -- celle-ci appelle le modele en direct
(predire_charge_globale() dans l'app), toujours a jour. Ce script sert la
page Fiabilite (courbe reel vs predit), qui a besoin d'etre rafraichie
separement pour couvrir les jours les plus recents.

Prerequis : models/model_charge_etp.pkl deja genere (modelisation_charge_etp.py).

Usage :
    python predict_charge.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))
from db_connection import get_engine, MODELS_DIR

from modelisation_charge_etp import TARGET, ajouter_features_cycliques
from modelisation_common import calculer_features_incidents

engine = get_engine()


def run():
    print("=" * 60)
    print("PREDICT - BLOC 3B : CHARGE PAR ETP (backtest)")
    print("=" * 60)

    print("\nChargement du modele entraine (models/model_charge_etp.pkl)...")
    modele = joblib.load(MODELS_DIR / "model_charge_etp.pkl")
    print(f"  Modele retenu : {modele['modele_retenu']} (MAE test={modele['mae_test']:.2f})")

    print("\nChargement de features_journalier...")
    df_jour = pd.read_sql("SELECT * FROM features_journalier", engine, parse_dates=["date"])
    df_jour = df_jour.sort_values("date").reset_index(drop=True)
    df_jour = ajouter_features_cycliques(df_jour)
    df_jour = df_jour.merge(calculer_features_incidents(df_jour["date"], engine), on="date", how="left")

    df_model = df_jour[modele["features"] + [TARGET, "date"]].dropna().copy()
    print(f"  Lignes utilisables : {len(df_model)} / {len(df_jour)}")

    X = df_model[modele["features"]]
    if modele["scaler"] is not None:
        X = modele["scaler"].transform(X)
    pred = modele["model"].predict(X)

    print("\nEtiquetage train/validation/test (dates de coupure figees a l'entrainement)...")
    date_fin_train = modele["date_fin_train"]
    date_fin_val = modele["date_fin_val"]
    split = pd.Series("test", index=df_model.index)
    split[df_model["date"] <= date_fin_val] = "validation"
    split[df_model["date"] <= date_fin_train] = "train"
    print(f"  train={(split == 'train').sum()}  validation={(split == 'validation').sum()}  "
          f"test={(split == 'test').sum()}")

    df_predictions = pd.DataFrame({
        "date": df_model["date"].values,
        "split": split.values,
        "charge_reelle": df_model[TARGET].values,
        "charge_predite": pred,
        "modele": modele["modele_retenu"],
    })

    print("\nEcriture dans PostgreSQL (predictions_charge_etp)...")
    df_predictions.to_sql("predictions_charge_etp", engine, if_exists="replace", index=False)
    print(f"  predictions_charge_etp : {len(df_predictions)} lignes ecrites")

    masque_test = split.values == "test"
    from sklearn.metrics import mean_absolute_error
    mae_test = mean_absolute_error(df_predictions.loc[masque_test, "charge_reelle"],
                                    df_predictions.loc[masque_test, "charge_predite"])
    print(f"  MAE recalcule sur le split test uniquement : {mae_test:.2f} "
          f"(reference entrainement : {modele['mae_test']:.2f})")

    print("\nPredict Bloc 3B termine.")


if __name__ == "__main__":
    run()
