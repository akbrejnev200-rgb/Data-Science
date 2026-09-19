# Détection AML — scoring de risque sur transactions bancaires

[![CI](https://github.com/akbrejnev200-rgb/Data-Science/actions/workflows/aml-ci.yml/badge.svg)](https://github.com/akbrejnev200-rgb/Data-Science/actions/workflows/aml-ci.yml)
[![CD](https://github.com/akbrejnev200-rgb/Data-Science/actions/workflows/aml-cd.yml/badge.svg)](https://github.com/akbrejnev200-rgb/Data-Science/actions/workflows/aml-cd.yml)

Pipeline de détection de blanchiment d'argent (AML) construit sur Google Cloud
Platform, combinant détection d'anomalies non supervisée et scoring de risque
supervisé, avec calibration de seuil orientée métier — et son déploiement
complet en MLOps sur Vertex AI (entraînement conteneurisé, pipeline avec
enregistrement conditionnel, Model Registry, app de scoring sur Cloud Run).

**Démo en ligne :** https://aml-suspect-scoring-demo-69646548838.us-central1.run.app
(scale-to-zero — la première requête peut prendre quelques secondes)

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
- **Random Forest / XGBoost** (supervisé) : scoring de risque par compte, sélectionnés
  sur un jeu de validation dédié, avec gestion explicite du déséquilibre de classes
- **Calibration de seuil** : règle métier (rappel cible sur validation), pas le
  seuil par défaut de 0,5 ni un score F arbitraire
- **Feature engineering** : agrégation des comportements d'envoi et de réception
  par compte (montants, diversité des contreparties, vélocité temporelle, ratios)

## Stack technique

Google Cloud Platform (BigQuery, Cloud Storage, Artifact Registry, Cloud Build,
Vertex AI Training/Pipelines/Model Registry, Cloud Run), Python (scikit-learn,
xgboost, pandas, FastAPI, KFP), Docker, SQL.

## Architecture MLOps

```
BigQuery (features)
      │
      ▼
┌───────────────────────────┐
│  Vertex AI Pipeline (KFP)  │
│                            │
│  train ──▶ auc_pr ≥ seuil ?│
│         ├─ oui → register  │──▶ Vertex AI Model Registry (versions + métriques)
│         └─ non → rapport   │           │
└───────────────────────────┘            │
                                          ▼
                              Cloud Run — app de scoring (FastAPI)
```

- **train** tourne dans l'image `trainer` (Custom Training Job ou composant de
  pipeline), publie modèle + métriques sur GCS
- **register** n'enregistre une nouvelle version que si le modèle dépasse un
  seuil d'AUC-PR — garde-fou contre la régression silencieuse
- L'**app de démo** charge la version enregistrée depuis GCS au démarrage :
  code et modèle sont déployés indépendamment

## CI/CD

Deux workflows GitHub Actions, déclenchés uniquement quand ce dossier change
(le repo est un monorepo — voir note plus bas), sans aucune clé/secret stocké :

- **CI** (`aml-ci.yml`) : sur chaque push/PR → lint (`ruff`) + tests (`pytest`)
- **CD** (`aml-cd.yml`) : sur push vers `main` (code de `serving/` ou `src/`) →
  rebuild l'image de démo, redéploie Cloud Run automatiquement

L'authentification GCP se fait par **Workload Identity Federation** : GitHub
prouve son identité via OIDC, Google échange ça contre un jeton court terme
pour un compte de service dédié (`github-deployer`, droits limités au build/
déploiement — pas les droits larges du compte utilisé pour l'entraînement).
Aucune clé de compte de service à faire fuiter.

## Architecture du projet

> Ce projet vit dans `06-MLOps-Vertex-AI/`, un sous-dossier du monorepo
> [Data-Science](https://github.com/akbrejnev200-rgb/Data-Science). Les workflows
> GitHub Actions (`.github/workflows/`) sont donc à la racine du repo, pas ici.

```
06-MLOps-Vertex-AI/
├── src/aml_detection/       # Package Python installable (pipeline d'entraînement)
│   ├── config.py            # Configuration centrale (env AML_*, défauts)
│   ├── data_loading.py      # Chargement des données depuis BigQuery
│   ├── features.py          # Définition et préparation des features
│   ├── train.py             # Entraînement des modèles
│   ├── evaluate.py          # Évaluation et calibration du seuil
│   ├── artifacts.py         # Publication des artefacts vers GCS
│   ├── registry.py          # Enregistrement dans le Model Registry (CLI + pipeline)
│   ├── batch_score.py       # Scoring par lots -> BigQuery (account_risk_scores)
│   └── main.py              # Orchestration du pipeline complet
├── pipelines/                # Pipeline Vertex AI (KFP)
│   ├── training_pipeline.py # Composants train / register / report_rejection
│   ├── training_pipeline.json  # Pipeline compilé (exécuté par Vertex)
│   └── run_pipeline.py      # Compilation + soumission d'un run
├── deploy/                   # Scripts MLOps ponctuels
│   ├── custom_job.yaml      # Spéc. du Custom Training Job
│   ├── batch_score_job.yaml # Spéc. du job de scoring par lots
│   └── register_model.py    # CLI d'enregistrement (utilise aml_detection.registry)
├── serving/                   # App de démo (Cloud Run)
│   ├── app.py                # API FastAPI (/predict, /health, page HTML)
│   ├── static/index.html    # Formulaire de test + exemples réels
│   ├── requirements.txt     # Dépendances minimales (image de serving allégée)
│   ├── Dockerfile
│   └── deploy.sh             # Build + déploiement Cloud Run
├── Dockerfile                 # Image d'entraînement
├── tests/                     # Tests unitaires (features, seuil, registry, config)
├── check_importance.py       # Diagnostic : importance des features
├── models/                    # Artefacts entraînés localement (non versionnés)
├── pyproject.toml
├── requirements-lock.txt     # Versions figées de l'image d'entraînement
└── README.md
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -e ".[mlops,dev]"     # cœur + SDK Vertex/KFP (deploy/, pipelines/) + outils de test
```

## Utilisation

```bash
# Authentification GCP (une fois)
gcloud auth application-default login

# Pipeline complet en local : chargement, entraînement, calibration, évaluation, sauvegarde
python -m aml_detection

# Entraînement dans le cloud (Custom Training Job)
gcloud builds submit --tag us-central1-docker.pkg.dev/mlops-vertex-demo/aml-detection/trainer:v2 .
gcloud ai custom-jobs create --region=us-central1 --display-name=aml-train --config=deploy/custom_job.yaml

# Pipeline complet (entraînement + enregistrement conditionnel)
python pipelines/run_pipeline.py

# Enregistrer un modèle existant manuellement
python deploy/register_model.py --artifact-uri gs://.../model --min-auc-pr 0.08

# Scoring par lots de tous les comptes -> BigQuery (aml_detection.account_risk_scores)
python -m aml_detection.batch_score --model-artifact-uri gs://.../model
gcloud ai custom-jobs create --region=us-central1 --display-name=aml-batch-score --config=deploy/batch_score_job.yaml

# App de démo
bash serving/deploy.sh
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

Modèle retenu : **XGBoost** (`max_depth=6`, `learning_rate=0.1`), sélectionné sur
validation face à un Random Forest (AUC-PR validation 0,101 vs 0,053).

| Métrique (jeu de test) | Valeur |
|---|---|
| AUC-ROC | 0,851 |
| AUC-PR (base rate 0,0075) | 0,093 |
| Seuil de décision (rappel cible 80 % sur validation) | 0,648 |
| Précision / rappel au seuil déployé | 0,052 / 0,476 |

Le modèle classe correctement (AUC-ROC élevé) mais la précision reste faible :
le jeu de features actuel ne sépare pas assez nettement les deux classes pour un
usage en production sans faux positifs massifs. Prochaine piste : enrichir les
features avec la structure du graphe de transactions (`feature_engineering_graph.sql`).

## Pistes d'évolution

- Model card et documentation de gouvernance
- Classification NLP des libellés de transaction
- Génération automatique de synthèses de risque (LLM)
