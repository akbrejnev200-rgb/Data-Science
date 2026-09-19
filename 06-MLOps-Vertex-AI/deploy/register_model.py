"""
CLI : enregistre un modèle entraîné dans le Vertex AI Model Registry.

Exemple :
    python deploy/register_model.py \
      --artifact-uri gs://mlops-vertex-demo-aml/jobs/aml-train-v1/model \
      --min-auc-pr 0.08

Le dossier `--artifact-uri` doit contenir `model.bst` et `metrics.json`.
La logique vit dans aml_detection.registry (partagée avec le composant du pipeline).
"""

import argparse

from aml_detection import config
from aml_detection.registry import register_version


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-uri", required=True,
                    help="gs://… dossier contenant model.bst et metrics.json")
    ap.add_argument("--min-auc-pr", type=float, default=0.0,
                    help="n'enregistre que si test.auc_pr >= ce seuil")
    args = ap.parse_args()

    register_version(
        config.PROJECT_ID, config.REGION, args.artifact_uri,
        min_auc_pr=args.min_auc_pr,
    )


if __name__ == "__main__":
    main()
