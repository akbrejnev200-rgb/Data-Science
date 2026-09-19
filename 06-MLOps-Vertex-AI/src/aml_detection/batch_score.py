"""
Scoring par lots : note tous les comptes d'account_features et écrit les
résultats dans BigQuery. Le pattern « batch » — facturé seulement pendant
l'exécution, pas de service allumé en permanence.

Réutilise l'image trainer (mêmes dépendances, même version de xgboost que
celle qui a produit le modèle — pas de risque de format entre versions).

    python -m aml_detection.batch_score --model-artifact-uri gs://.../model
"""

import argparse
import datetime as dt
from pathlib import Path

import joblib
from google.cloud import bigquery, storage

from .config import BQ_DATASET, PROJECT_ID
from .data_loading import load_account_features
from .features import FEATURES, prepare_features

OUTPUT_TABLE = "account_risk_scores"
LOCAL_DIR = Path("/tmp/batch_model")


def _download(gcs_uri: str, filename: str) -> Path:
    bucket_name, _, prefix = gcs_uri.removeprefix("gs://").partition("/")
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    dest = LOCAL_DIR / filename
    storage.Client(project=PROJECT_ID).bucket(bucket_name).blob(
        f"{prefix.rstrip('/')}/{filename}"
    ).download_to_filename(str(dest))
    return dest


def run(model_artifact_uri: str, model_version: str = "unknown") -> int:
    print(f"Chargement du modèle depuis {model_artifact_uri}")
    model = joblib.load(_download(model_artifact_uri, "best_model.joblib"))
    threshold = float(joblib.load(_download(model_artifact_uri, "decision_threshold.joblib")))

    print("Chargement des comptes depuis BigQuery...")
    df = load_account_features()
    account_keys = df["account_key"].reset_index(drop=True)
    X, _ = prepare_features(df)

    print(f"Scoring de {len(X)} comptes...")
    proba = model.predict_proba(X[FEATURES])[:, 1]

    result = account_keys.to_frame(name="account_key")
    result["probability"] = proba.round(4)
    result["verdict"] = ["suspect" if p >= threshold else "normal" for p in proba]
    result["decision_threshold"] = threshold
    result["model_version"] = model_version
    result["scored_at"] = dt.datetime.now(dt.timezone.utc)

    table_id = f"{PROJECT_ID}.{BQ_DATASET}.{OUTPUT_TABLE}"
    print(f"Écriture de {len(result)} lignes dans {table_id} (WRITE_TRUNCATE)...")
    job = bigquery.Client(project=PROJECT_ID).load_table_from_dataframe(
        result, table_id,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    )
    job.result()

    n_suspect = int((result["verdict"] == "suspect").sum())
    print(f"Terminé. {n_suspect} comptes suspects sur {len(result)} "
          f"({n_suspect / len(result) * 100:.2f}%).")
    return n_suspect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-artifact-uri", required=True,
                    help="gs://… dossier contenant best_model.joblib et decision_threshold.joblib")
    ap.add_argument("--model-version", default="unknown",
                    help="étiquette de traçabilité écrite dans la table de sortie")
    args = ap.parse_args()
    run(args.model_artifact_uri, args.model_version)


if __name__ == "__main__":
    main()
