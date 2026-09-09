"""
Configuration centrale du projet.

Toutes les valeurs sont lues depuis l'environnement (préfixe ``AML_``) avec un
défaut raisonnable, pour que le même code tourne en local, dans un conteneur ou
dans un composant Vertex AI sans modification.
"""

import os
from pathlib import Path

# --- GCP ---
PROJECT_ID = os.environ.get("AML_PROJECT_ID", "mlops-vertex-demo")
REGION = os.environ.get("AML_REGION", "us-central1")

# --- BigQuery (source des features) ---
BQ_DATASET = os.environ.get("AML_BQ_DATASET", "aml_detection")
BQ_TABLE = os.environ.get("AML_BQ_TABLE", "account_features")

# --- Stockage des artefacts ---
# Racine du repo = deux niveaux au-dessus de src/aml_detection/
REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("AML_MODELS_DIR", str(REPO_ROOT / "models")))

# Bucket GCS de staging (artefacts, pipelines Vertex). Défini plus tard, à
# l'étape conteneurisation ; vide tant qu'on travaille en local.
GCS_BUCKET = os.environ.get("AML_GCS_BUCKET", "")

# Destination GCS des artefacts d'entraînement (modèles + metrics.json).
# - vide  -> on écrit seulement en local (MODELS_DIR), pour les runs sur PC
# - gs://… -> on pousse aussi les fichiers là-bas après la sauvegarde locale
# AIP_MODEL_DIR est renseigné automatiquement par Vertex AI Custom Training :
# un job cloud publie donc ses artefacts sans configuration supplémentaire.
ARTIFACTS_URI = os.environ.get("AML_ARTIFACTS_URI", os.environ.get("AIP_MODEL_DIR", ""))

# --- Hyperparamètres métier ---
# Contamination a priori de l'Isolation Forest : ordre de grandeur (~2 %) des
# comptes signalés suspects dans les portefeuilles AML réels. Indépendant des
# labels du jeu courant (contrairement à y.mean(), indisponible en production).
CONTAMINATION_PRIOR = float(os.environ.get("AML_CONTAMINATION_PRIOR", "0.02"))

# Rappel minimal exigé sur la classe suspecte pour fixer le seuil de décision
# (règle métier / conformité).
TARGET_RECALL = float(os.environ.get("AML_TARGET_RECALL", "0.80"))
