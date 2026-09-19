# Model Card — aml-suspect-scoring

Documentation du modèle au sens des recommandations de gouvernance ML (inspirée
du format [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
et du Model Registry Vertex AI). À lire avant toute décision d'usage.

## Détails du modèle

| | |
|---|---|
| Nom | `aml-suspect-scoring` |
| Version courante | v2 |
| Type | Classification binaire supervisée |
| Algorithme | XGBoost (`XGBClassifier`), `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`, `scale_pos_weight` équilibré |
| Framework | xgboost 3.2.0, scikit-learn 1.7.2 |
| Sélectionné face à | Random Forest (AUC-PR validation 0,101 vs 0,053) et Isolation Forest (non supervisé, référence : rappel 4,6 %) |
| Produit par | Vertex AI Pipeline `aml-training`, run `aml-training-20260911004231`, 2026-09-11 |
| Registre | Vertex AI Model Registry, projet `mlops-vertex-demo`, région `us-central1` |
| Auteur / contact | Koffi Brejnev Akoumani |
| Licence des données | Dataset public de recherche IBM AML (usage académique/démo) |

## Usage prévu

**Usage visé** : projet pédagogique et démonstration de bout en bout d'un
pipeline MLOps (entraînement conteneurisé, orchestration, Model Registry,
serving, CI/CD) sur un cas d'usage AML. Le score produit illustre une
**priorisation d'alertes pour revue humaine**, pas une décision automatisée.

**Hors périmètre — à ne pas faire** :
- Déploiement en production sur des données bancaires réelles
- Toute décision réglementaire, légale ou de blocage de compte fondée
  uniquement sur ce score
- Usage sans supervision humaine sur les comptes signalés

## Données

- **Source** : dataset public de recherche [IBM AML](https://github.com/IBM/AMLSim)
  — transactions synthétiques générées par simulation multi-agents, aucune
  donnée personnelle ou bancaire réelle.
- **Feature engineering** : agrégation par compte (montants, diversité des
  contreparties et devises, vélocité temporelle, réciprocité des échanges —
  voir `feature_engineering_graph.sql`), 19 features numériques.
- **Volume** : 705 907 comptes, dont 5 304 étiquetés suspects (**0,75 %**) —
  déséquilibre représentatif du sujet (la fraude réelle est rare).
- **Split** : stratifié 70 / 15 / 15 (train / validation / test), le jeu de
  test n'est utilisé qu'une seule fois, en toute fin de pipeline.
- **Biais connu des données** : les schémas de blanchiment sont *simulés*
  selon des règles connues des concepteurs du dataset — un modèle qui
  performe bien ici n'est **pas** une garantie de performance sur des
  techniques réelles, potentiellement plus subtiles ou adversariales.

## Métriques (jeu de test, jamais vu avant l'évaluation finale)

| Métrique | Valeur |
|---|---|
| AUC-ROC | 0,851 |
| AUC-PR (base rate 0,75 %) | 0,093 |
| Seuil de décision déployé | 0,648 |
| Rappel visé (règle métier, sur validation) | 80 % |
| Rappel réel (test) | 47,6 % |
| Précision (test) | 5,2 % |

Sur l'ensemble des 705 907 comptes (scoring par lots) : **48 260 signalés
suspects (6,8 %)**, avec la même précision d'environ 5 % — soit ~19 fausses
alertes pour 1 vrai cas retrouvé.

## Considérations éthiques et limites

- **Précision faible = coût opérationnel réel.** À grande échelle, ce modèle
  génère un volume de fausses alertes qui rendrait une revue manuelle
  exhaustive impraticable sans triage humain supplémentaire.
- **Faux négatifs.** Le modèle ne retrouve que ~48 % des cas suspects du jeu
  de test. En contexte réel, un faux négatif AML a des conséquences bien plus
  graves qu'un faux positif — c'est pourquoi le seuil est calibré pour
  favoriser le rappel (voir Méthodologie du README), mais 48 % reste
  insuffisant pour un déploiement réel.
- **Biais indirect.** Aucune feature démographique n'est utilisée
  volontairement. Le risque de biais par proxy reste possible : des features
  comme le volume de transactions ou le nombre de contreparties peuvent
  corréler avec la taille ou le secteur d'activité d'un compte (ex. comptes
  professionnels vs particuliers), sans que cela ait été audité ici.
- **Dérive non surveillée.** Aucun monitoring de dérive (data/concept drift)
  n'est en place sur le modèle déployé — à ajouter avant tout usage prolongé
  (Vertex AI Model Monitoring, piste d'évolution).
- **Origine synthétique.** Voir biais des données ci-dessus : le principal
  facteur limitant la validité du modèle hors de ce projet.

## Recommandations

1. **Ne jamais utiliser seul** : ce score doit alimenter une revue humaine,
   pas remplacer une décision de conformité.
2. **Réentraîner régulièrement** — le pipeline (`pipelines/run_pipeline.py`)
   n'enregistre une nouvelle version que si elle dépasse le seuil d'AUC-PR
   (0,08), garde-fou contre une régression silencieuse.
3. **Prochaine piste d'amélioration prioritaire** : intégrer des features de
   structure du graphe de transactions (chaînes, cycles, communautés) —
   `feature_engineering_graph.sql` pose déjà les bases côté SQL. Les features
   actuelles (agrégats par compte) plafonnent la précision quel que soit le
   seuil retenu ou l'algorithme choisi.
4. **Avant tout usage réel** : audit de biais formel, monitoring de dérive,
   revue réglementaire/légale (hors périmètre de ce projet démo).

## Reproductibilité

- Pipeline complet : `pipelines/training_pipeline.py` (KFP), run reproductible
  via `python pipelines/run_pipeline.py`
- Artefacts du modèle : `gs://mlops-vertex-demo-aml/pipeline-root/…/model/`
  (`model.bst`, `metrics.json`)
- Historique des versions : Vertex AI Model Registry → `aml-suspect-scoring`
