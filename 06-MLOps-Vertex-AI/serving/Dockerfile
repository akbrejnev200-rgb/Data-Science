# Image de serving : API de démo (FastAPI) pour le modèle AML.
# Séparée de l'image d'entraînement (Dockerfile à la racine) : dépendances
# minimales -> image plus légère, démarrage à froid plus rapide sur Cloud Run.
#
# Build (depuis la racine du repo, car le contexte inclut src/) :
#   gcloud builds submit --config serving/cloudbuild.yaml .

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY serving/requirements.txt ./serving-requirements.txt
# nvidia-nccl-cu12 est tiré par xgboost (communication multi-GPU) mais inutile
# en inference CPU : ~340 Mo. Installé puis retiré dans la même couche pour ne
# pas alourdir l'image (sinon il resterait dans l'historique des couches).
RUN pip install --no-cache-dir -r serving-requirements.txt \
    && pip uninstall -y --no-input nvidia-nccl-cu12

# Seul aml_detection.features (liste FEATURES) est utilisé ici ; --no-deps
# évite de réinstaller pandas/bigquery/etc. déjà exclus ci-dessus.
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

COPY serving/ ./serving/

EXPOSE 8080
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
