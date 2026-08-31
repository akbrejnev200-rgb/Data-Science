"""
predict_retard.py

PREDICT UNIQUEMENT pour le Bloc 3C : charge le modele deja entraine
(models/model_classification_retard.pkl) et (re)ecrit le backtest
predictions_risque_retard, avec une colonne "split" (train/validation/
test) absente de l'ancienne version -- sans elle, la matrice de confusion
affichee sur la page Fiabilite melangeait des predictions sur des donnees
deja vues a l'entrainement avec des donnees jamais vues, ce qui la rendait
plus optimiste qu'elle ne devrait l'etre.

Toute ligne posterieure a la date de fin de validation d'origine (stockee
dans le modele) est etiquetee "test" -- y compris les taches closes depuis
le dernier entrainement, par definition jamais vues par le modele.

Prerequis : models/model_classification_retard.pkl deja genere
(modelisation_retard.py).

Usage :
    python predict_retard.py
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
    print("PREDICT - BLOC 3C : RISQUE DE RETARD (backtest)")
    print("=" * 60)

    print("\nChargement du modele entraine (models/model_classification_retard.pkl)...")
    modele = joblib.load(MODELS_DIR / "model_classification_retard.pkl")
    print(f"  Modele retenu : {modele['type']} (accuracy test={modele['accuracy_test']:.3f})")
    print(f"  Seuil de decision : {modele['seuil_decision']:.3f}")

    print("\nChargement de features_detail...")
    df = pd.read_sql("SELECT * FROM features_detail", engine, parse_dates=["Date_Creation"])
    print(f"  {len(df)} lignes")

    print("\nConstruction de la cible (seuil fige a l'entrainement)...")
    seuil_retard_j = modele["seuil_retard_j"]
    df["est_en_retard"] = (df["duree_traitement_reelle_j"] > seuil_retard_j).astype(int)
    df_clf = df[df["duree_traitement_reelle_j"].notna()].copy()
    print(f"  Lignes avec duree connue (taches closes) : {len(df_clf)} / {len(df)}")

    df_clf["Service_ref"] = df_clf["Service"]
    df_clf = pd.get_dummies(df_clf, columns=["Service", "Type_Contrat", "Type_Processus"],
                             prefix=["svc", "ctr", "proc"])

    cols_id = ["ID_Tache", "Matricule_Agent", "Date_Creation", "Service_ref"]
    colonnes_dispo = [c for c in modele["features"] if c in df_clf.columns]
    df_model = df_clf[colonnes_dispo + ["est_en_retard"] + cols_id].dropna()
    df_model = df_model.sort_values("Date_Creation").reset_index(drop=True)
    print(f"  Lignes utilisables (features completes) : {len(df_model)}")

    X = df_model.reindex(columns=modele["features"], fill_value=0)

    print("\nScoring de l'historique complet...")
    proba = modele["model"].predict_proba(X)[:, 1]
    pred = (proba >= modele["seuil_decision"]).astype(int)

    print("\nEtiquetage train/validation/test (dates de coupure figees a l'entrainement)...")
    date_fin_train = modele["date_fin_train"]
    date_fin_val = modele["date_fin_val"]
    split = pd.Series("test", index=df_model.index)
    split[df_model["Date_Creation"] <= date_fin_val] = "validation"
    split[df_model["Date_Creation"] <= date_fin_train] = "train"
    print(f"  train={(split == 'train').sum()}  validation={(split == 'validation').sum()}  "
          f"test={(split == 'test').sum()}")

    df_predictions = df_model[cols_id].rename(columns={"Service_ref": "Service"}).copy()
    df_predictions["score_risque_retard"] = proba
    df_predictions["est_en_retard_predit"] = pred
    df_predictions["est_en_retard_reel"] = df_model["est_en_retard"].values
    df_predictions["split"] = split.values

    print("\nEcriture dans PostgreSQL (predictions_risque_retard)...")
    df_predictions.to_sql("predictions_risque_retard", engine, if_exists="replace", index=False, chunksize=5000)
    print(f"  predictions_risque_retard : {len(df_predictions)} lignes ecrites")

    masque_test = split.values == "test"
    acc_test = (pred[masque_test] == df_model.loc[masque_test, "est_en_retard"].values).mean()
    print(f"  Accuracy recalculee sur le split test uniquement : {acc_test:.3f} "
          f"(reference entrainement : {modele['accuracy_test']:.3f})")

    print("\nPredict Bloc 3C termine.")


if __name__ == "__main__":
    run()
