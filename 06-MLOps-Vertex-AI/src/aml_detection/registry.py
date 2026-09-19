"""
Enregistrement d'une version de modèle dans le Vertex AI Model Registry.

Logique partagée par la CLI (``deploy/register_model.py``) et le composant du
pipeline. Le SDK ``google-cloud-aiplatform`` est importé à l'usage : le package
reste installable sans lui (l'image d'entraînement ne l'embarque pas).
"""

import json

MODEL_DISPLAY_NAME = "aml-suspect-scoring"
SERVING_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"


def _parse_gcs(uri: str) -> tuple[str, str]:
    bucket, _, prefix = uri.removeprefix("gs://").partition("/")
    return bucket, prefix.strip("/")


def _label(value: float) -> str:
    """Une métrique -> une étiquette GCP valide (minuscules, chiffres, _ et -)."""
    return f"{value:.3f}".replace(".", "_")


def register_version(
    project_id: str,
    region: str,
    artifact_uri: str,
    *,
    display_name: str = MODEL_DISPLAY_NAME,
    serving_image: str = SERVING_IMAGE,
    min_auc_pr: float = 0.0,
) -> str | None:
    """Enregistre `artifact_uri` (dossier GCS contenant model.bst + metrics.json)
    comme nouvelle version de `display_name`.

    Renvoie le resource name du modèle, ou None si l'AUC-PR test est sous
    `min_auc_pr` (modèle rejeté, rien n'est enregistré).
    """
    from google.cloud import aiplatform, storage

    artifact_uri = artifact_uri.rstrip("/")
    bucket, prefix = _parse_gcs(f"{artifact_uri}/metrics.json")
    metrics = json.loads(
        storage.Client(project=project_id).bucket(bucket).blob(prefix).download_as_text()
    )
    auc_pr = float(metrics["test"]["auc_pr"])

    if auc_pr < min_auc_pr:
        print(f"AUC-PR test = {auc_pr:.3f} < seuil {min_auc_pr:.3f} -> modèle rejeté.")
        return None

    aiplatform.init(project=project_id, location=region)
    existing = aiplatform.Model.list(filter=f'display_name="{display_name}"')
    parent = existing[0].resource_name if existing else None

    model = aiplatform.Model.upload(
        display_name=display_name,
        parent_model=parent,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_image,
        description=json.dumps(metrics),
        labels={
            "model": metrics["model_name"].lower(),
            "auc_pr": _label(auc_pr),
            "recall_suspect": _label(metrics["test"]["recall_suspect"]),
        },
    )
    kind = "nouvelle version" if parent else "premier enregistrement"
    print(f"Enregistré ({kind}) : {model.resource_name} — version {model.version_id}")
    return model.resource_name
