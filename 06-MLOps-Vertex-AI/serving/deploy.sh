#!/usr/bin/env bash
# Build + déploiement de l'app de démo sur Cloud Run.
# Usage (depuis la racine du repo) : bash serving/deploy.sh [model_artifact_uri]
set -euo pipefail

PROJECT_ID="mlops-vertex-demo"
REGION="us-central1"
IMAGE="us-central1-docker.pkg.dev/${PROJECT_ID}/aml-detection/serving:v2"
SERVICE="aml-suspect-scoring-demo"

# Par défaut : l'artefact du modèle enregistré (aml-suspect-scoring, dernière
# version acceptée par le pipeline d'entraînement).
MODEL_ARTIFACT_URI="${1:-gs://mlops-vertex-demo-aml/pipeline-root/69646548838/aml-training-20260911004231/train_6702289541920718848/model}"

gcloud builds submit --config serving/cloudbuild.yaml .

gcloud run deploy "$SERVICE" \
  --image="$IMAGE" \
  --region="$REGION" \
  --platform=managed \
  --allow-unauthenticated \
  --port=8080 \
  --memory=512Mi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=2 \
  --set-env-vars="MODEL_ARTIFACT_URI=${MODEL_ARTIFACT_URI}"
