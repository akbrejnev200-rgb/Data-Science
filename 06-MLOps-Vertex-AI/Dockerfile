# Image d'entraînement du pipeline AML.
# Emballe le package aml_detection + ses dépendances figées pour un run
# reproductible en local comme sur Vertex AI Custom Training.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dépendances d'abord : tant que le lock ne change pas, Docker réutilise le
# cache de cette couche et ne réinstalle rien au build suivant.
COPY pyproject.toml requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

# Puis le code, et installation du package sans re-résoudre les versions.
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

# Au démarrage du conteneur : lance le pipeline complet.
ENTRYPOINT ["python", "-m", "aml_detection"]
