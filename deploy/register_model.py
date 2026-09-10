"""
Enregistre le modèle entraîné dans le Vertex AI Model Registry.

Lit ``metrics.json`` à la racine de ``--source-uri``, copie ``model.bst`` +
``metrics.json`` vers ``--model-uri`` (emplacement propre de l'artefact du modèle
enregistré), puis crée une version du modèle « aml-suspect-scoring ».

``--min-auc-pr`` : n'enregistre que si l'AUC-PR test dépasse ce seuil. Utile
quand le pipeline (étape 3) appelle ce script : on ne pollue pas le registre
avec un modèle qui a régressé.

Exemple :
    python -m deploy.register_model \
      --source-uri gs://mlops-vertex-demo-aml/jobs/aml-train-v1/model \
      --model-uri  gs://mlops-vertex-demo-aml/models/aml-suspect-scoring/v1 \
      --min-auc-pr 0.08
"""

import argparse
import json

from google.cloud import aiplatform, storage

from aml_detection import config

MODEL_DISPLAY_NAME = "aml-suspect-scoring"
SERVING_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"


def _parse_gcs(uri: str) -> tuple[str, str]:
    bucket, _, prefix = uri.removeprefix("gs://").partition("/")
    return bucket, prefix.strip("/")


def _read_json(client: storage.Client, uri: str) -> dict:
    bucket, blob = _parse_gcs(uri)
    return json.loads(client.bucket(bucket).blob(blob).download_as_text())


def _copy(client: storage.Client, src_uri: str, dst_uri: str) -> None:
    sb, sblob = _parse_gcs(src_uri)
    db, dblob = _parse_gcs(dst_uri)
    src_bucket = client.bucket(sb)
    src_bucket.copy_blob(src_bucket.blob(sblob), client.bucket(db), dblob)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-uri", required=True,
                    help="gs://… dossier produit par le job (model.bst, metrics.json)")
    ap.add_argument("--model-uri", required=True,
                    help="gs://… dossier propre pour l'artefact enregistré")
    ap.add_argument("--min-auc-pr", type=float, default=0.0,
                    help="n'enregistre que si test.auc_pr >= ce seuil")
    args = ap.parse_args()

    source = args.source_uri.rstrip("/")
    target = args.model_uri.rstrip("/")

    gcs = storage.Client(project=config.PROJECT_ID)
    metrics = _read_json(gcs, f"{source}/metrics.json")
    auc_pr = metrics["test"]["auc_pr"]

    if auc_pr < args.min_auc_pr:
        print(f"AUC-PR test = {auc_pr:.3f} < seuil {args.min_auc_pr:.3f} "
              f"-> modèle non enregistré.")
        return

    for name in ("model.bst", "metrics.json"):
        _copy(gcs, f"{source}/{name}", f"{target}/{name}")
        print(f"copié : {target}/{name}")

    aiplatform.init(project=config.PROJECT_ID, location=config.REGION)

    existing = aiplatform.Model.list(filter=f'display_name="{MODEL_DISPLAY_NAME}"')
    parent = existing[0].resource_name if existing else None

    def _lbl(x: float) -> str:
        return f"{x:.3f}".replace(".", "_")

    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        parent_model=parent,
        artifact_uri=target,
        serving_container_image_uri=SERVING_IMAGE,
        description=json.dumps(metrics),
        labels={
            "model": metrics["model_name"].lower(),
            "auc_pr": _lbl(auc_pr),
            "recall_suspect": _lbl(metrics["test"]["recall_suspect"]),
        },
    )
    print(f"\nModèle enregistré : {model.resource_name}")
    print(f"Version : {model.version_id}  (nouvelle version d'un modèle existant)"
          if parent else f"Version : {model.version_id}  (premier enregistrement)")


if __name__ == "__main__":
    main()
