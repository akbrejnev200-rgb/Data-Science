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
mlops_project/
├── src/
│   ├── data_loading.py   # Chargement des données depuis BigQuery
│   ├── features.py        # Définition et préparation des features
│   ├── train.py            # Entraînement des modèles
│   ├── evaluate.py         # Évaluation et calibration du seuil
│   └── main.py              # Orchestration du pipeline complet
├── models/                  # Modèles entraînés (.joblib, non versionnés)
├── requirements.txt
└── README.md
```

## Utilisation

```bash
cd src
python main.py
```

## Méthodologie

- Split stratifié 70/15/15 (train/validation/test)
- Sélection d'hyperparamètre (profondeur d'arbre) sur le jeu de validation
- Calibration du seuil de décision par optimisation du F1-score sur validation
- Évaluation finale unique sur le jeu de test, jamais utilisé avant cette étape

## Résultats clés

À compléter au fil des itérations (AUC-ROC, AUC-PR, rappel/précision par seuil).

## Pistes d'évolution

- Classification NLP des libellés de transaction
- Génération automatique de synthèses de risque (LLM)
- Déploiement en production via Vertex AI Pipelines
- Documentation de gouvernance modèle (model card)
