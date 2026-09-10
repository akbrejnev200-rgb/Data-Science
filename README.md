# Détection AML — Transactions suspectes (Datalab RISQ)

# Détection AML — Pipeline de scoring de risque sur transactions bancaires

Pipeline de détection de blanchiment d'argent (AML) construit sur Google Cloud
Platform, combinant détection d'anomalies non supervisée et scoring de risque
supervisé, avec calibration de seuil orientée métier.

## Contexte

Le blanchiment d'argent représente un défi majeur pour les institutions
financières : les schémas sont volontairement conçus pour se fondre dans le
volume de transactions légitimes, et les jeux de données sont extrêmement
déséquilibrés (souvent moins de 1% de cas positifs). Ce projet explore une
approche de bout en bout, du stockage cloud des données jusqu'au scoring de
risque par compte, en s'appuyant sur le dataset public de recherche IBM AML
(transactions synthétiques générées par un simulateur multi-agents,
largement utilisé dans la littérature académique sur la détection AML et le
graph learning).

## Approche technique

- **Isolation Forest** (non supervisé) : détection d'anomalies sans a priori sur les labels
- **Random Forest** (supervisé) : scoring de risque par compte, avec gestion explicite du déséquilibre de classes
- **Calibration de seuil** : optimisation du compromis précision/rappel sur un jeu de validation dédié, plutôt que d'utiliser le seuil par défaut de 0.5
- **Feature engineering** : agrégation des comportements d'envoi et de réception par compte (montants, diversité des contreparties, vélocité temporelle, ratios)

## Stack technique

Google Cloud Platform (BigQuery, Cloud Storage, Vertex AI), Python
(scikit-learn, pandas), SQL.

## Architecture du projet

```
mlops-vertex-demo/
├── src/aml_detection/       # Package Python installable (pipeline d'entraînement)
│   ├── config.py            # Configuration centrale (env AML_*, défauts)
│   ├── data_loading.py      # Chargement des données depuis BigQuery
│   ├── features.py          # Définition et préparation des features
│   ├── train.py             # Entraînement des modèles
│   ├── evaluate.py          # Évaluation et calibration du seuil
│   ├── artifacts.py         # Publication des artefacts vers GCS
│   └── main.py              # Orchestration du pipeline complet
├── deploy/                  # Scripts MLOps (Model Registry, jobs Vertex)
│   ├── custom_job.yaml      # Spéc. du Custom Training Job
│   └── register_model.py    # Enregistrement dans le Vertex AI Model Registry
├── Dockerfile               # Image d'entraînement
├── check_importance.py      # Diagnostic : importance des features
├── models/                  # Artefacts entraînés (non versionnés)
├── pyproject.toml
├── requirements-lock.txt    # Versions figées de l'image d'entraînement
└── README.md
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -e ".[mlops,dev]"     # cœur + SDK Vertex (deploy/) + outils de test
```

## Utilisation

```bash
# Authentification GCP (une fois)
gcloud auth application-default login

# Pipeline complet : chargement, entraînement, calibration, évaluation, sauvegarde
python -m aml_detection
```

La configuration (projet GCP, région, dataset BigQuery, rappel cible…) se surcharge
par variables d'environnement `AML_*` — voir `src/aml_detection/config.py`.

## Méthodologie

- Split stratifié 70/15/15 (train/validation/test)
- Sélection d'hyperparamètre **et** choix du modèle (RF vs XGBoost) sur le jeu de validation
- Scaler ajusté sur le train uniquement ; Isolation Forest entraîné sur le train,
  contamination fixée a priori (pas de fuite de label)
- Seuil de décision fixé par une **règle métier** : le seuil le plus précis
  atteignant un rappel cible sur la classe suspecte (validation)
- Évaluation finale unique sur le jeu de test, du seul modèle retenu

## Résultats clés

À compléter au fil des itérations (AUC-ROC, AUC-PR, rappel/précision par seuil).

## Pistes d'évolution

- Classification NLP des libellés de transaction
- Génération automatique de synthèses de risque (LLM)
- Déploiement en production via Vertex AI Pipelines
- Documentation de gouvernance modèle (model card)
