"""
Publication des artefacts d'entraînement vers Google Cloud Storage.

Utilisé quand ``config.ARTIFACTS_URI`` pointe sur un ``gs://…`` (job Vertex ou
run local avec ``AML_ARTIFACTS_URI``). Sans ça, le pipeline reste 100 % local.
"""

from pathlib import Path

from google.cloud import storage

GCS_PREFIX = "gs://"


def _split_uri(gcs_uri: str) -> tuple[str, str]:
    """gs://bucket/chemin/vers/dossier -> ("bucket", "chemin/vers/dossier")."""
    if not gcs_uri.startswith(GCS_PREFIX):
        raise ValueError(f"URI GCS invalide (attendu gs://…) : {gcs_uri!r}")
    bucket_name, _, prefix = gcs_uri[len(GCS_PREFIX):].partition("/")
    return bucket_name, prefix.strip("/")


def upload_dir(local_dir, gcs_uri: str) -> list[str]:
    """Copie récursivement le contenu de `local_dir` vers `gcs_uri`.

    Renvoie la liste des URI GCS écrites.
    """
    bucket_name, prefix = _split_uri(gcs_uri)
    bucket = storage.Client().bucket(bucket_name)
    local_dir = Path(local_dir)

    written = []
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        blob_name = f"{prefix}/{rel}" if prefix else rel
        bucket.blob(blob_name).upload_from_filename(str(path))
        written.append(f"{GCS_PREFIX}{bucket_name}/{blob_name}")
    return written
