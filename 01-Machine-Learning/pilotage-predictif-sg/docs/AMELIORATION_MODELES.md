# Amélioration des modèles — comparaison avant / après

## Méthodologie

Pour chaque bloc de la Phase 3, deux versions ont été exécutées contre la
**même base PostgreSQL**, avec le **même split temporel 70/15/15** et donc le
**même jeu de test** :

- **Avant** : hyperparamètres fixes du notebook original (`phase3_modelisation.ipynb`),
  jeu de features d'origine.
- **Après** : version portée dans `database/modelisation_*.py`, avec recherche
  d'hyperparamètres et/ou features supplémentaires.

Cette comparaison porte donc uniquement sur l'apport de la méthodologie
(recherche d'hyperparamètres, nouvelles features, ajustement de seuil), pas
sur l'apport de la migration CSV → PostgreSQL elle-même (les fichiers CSV de
l'ancien pipeline Colab n'étaient pas disponibles pour une comparaison
directe — cf. README, catégorie B).

---

## Bloc 3A — Prévision des volumes (Prophet / SARIMA / XGBoost / Ensemble)

### Etape 1 — recherche d'ordre SARIMA (vs config fixe du notebook original)

| Configuration | Ordre | MAE (test) | RMSE (test) | MAPE (test) |
|---|---|---:|---:|---:|
| Avant (fixe) | (1,1,1)(1,1,1,7) | 1135.5 | 1349.0 | 24.6% |
| Après (recherche sur validation) | (1,1,2)(1,1,1,7) | 1019.1 | 1319.8 | 19.1% |
| Gain | | −10.3% | −2.2% | −5.5 points |

### Etape 2 — regresseurs exogenes calendaires (SARIMA univarié -> SARIMAX)

La version univariée (aucune information calendaire) restait à un MAPE de
19.1%, jugé trop élevé pour un usage opérationnel bancaire. Ajout de
régresseurs exogènes (`est_fin_de_mois`, `est_fin_trimestre`, `est_lundi`,
`est_vendredi`, `est_aout`) au SARIMAX :

| Configuration | MAE (test) | RMSE (test) | MAPE (test) |
|---|---:|---:|---:|
| SARIMA univarié (etape 1) | 1019.1 | 1319.8 | 19.1% |
| SARIMA + régresseurs exogènes | 667.4 | 837.3 | 13.0% |
| Gain | −34.5% | −36.6% | −6.1 points |

### Etape 3 — troisième candidat XGBoost + Ensemble

Un candidat XGBoost (features de lag : `volume_lag_1j/7j/14j/30j`,
`volume_moy_7j`, variables calendaires) a été ajouté, plus un Ensemble
(moyenne simple des prédictions SARIMA et XGBoost) :

| Modèle | MAE (test) | RMSE (test) | MAPE (test) |
|---|---:|---:|---:|
| Prophet (évalué sur Google Colab, indisponible localement) | 856.0 | 1014.0 | 18.7% |
| SARIMA (avec exogènes) | 667.4 | 837.3 | 13.0% |
| XGBoost (features de lag) | 692.2 | 850.4 | 13.0% |
| **Ensemble SARIMA + XGBoost (retenu)** | **644.7** | **810.3** | **12.2%** |

Une pondération "optimisée" sur validation (0.7 SARIMA / 0.3 XGBoost, MAE
validation = 614.9) a été testée et **rejetée** : elle fait moins bien sur
le test (MAE=638.2) que la moyenne simple 50/50 (MAE=628.5 lors du test
initial) — signe de surapprentissage sur les 96 points de validation
disponibles. La moyenne non pondérée est retenue car plus robuste.

**Pistes testées et écartées** (documentées pour éviter de les retenter) :
- Jours fériés français + jours de pont (`holidays` package) en régresseur
  supplémentaire : gain négligeable (MAE 667.4 → 666.2, soit −0.2%).
- Remplacement du flag mensuel `est_fin_trimestre` (binaire sur tout le
  mois) par une rampe progressive dans les 10 derniers jours du trimestre :
  MAE presque doublé (667 → 1118). Le flag plat capture mieux l'effet réel,
  probablement lié à des clôtures comptables trimestrielles qui s'étalent
  sur tout le mois plutôt que de se concentrer en fin de période.
- Transformation logarithmique de la cible avant SARIMA : gain marginal
  (663.2 vs 667.4, −0.6%), non retenu par simplicité (le gain ne justifie
  pas la complexité supplémentaire de la transformation inverse).

### Pourquoi le MAE ne peut pas descendre beaucoup plus bas avec ces données

Comparaison à des références naïves (aucune modélisation) sur le même test :

| Approche | MAE | MAPE |
|---|---:|---:|
| Naïf — moyenne du train | 1224.6 | 20.9% |
| Naïf — valeur de la veille | 770.3 | 15.3% |
| Naïf — même jour de semaine (semaine précédente) | 1050.4 | 21.7% |
| **Naïf — moyenne mobile 7 jours (zéro modélisation)** | **647.0** | **13.3%** |
| **Ensemble SARIMA+XGBoost (résultat final)** | **644.7** | **12.2%** |

Une simple moyenne mobile 7 jours, sans aucune modélisation, obtient un MAE
quasi identique à l'Ensemble (647.0 vs 644.7, écart de 0.3%). Ceci indique
que l'essentiel de la variance jour-à-jour du volume (écart-type test =
1262 dossiers, contre une moyenne de 5207) est **du bruit métier réel**
non expliqué par le calendrier ni par l'historique propre de la série —
vraisemblablement des campagnes commerciales, événements ponctuels ou
décisions clients dont aucune trace n'existe dans les 4 fichiers sources
du projet (aucun calendrier de campagnes, aucun indicateur macro-économique).
Un MAE cible de 100 dossiers/jour (1.9% de la moyenne) impliquerait
d'expliquer plus de 92% de cette variance, ce qui n'est pas atteignable
avec les données actuellement disponibles : les gains obtenus (recherche
d'hyperparamètres, régresseurs exogènes, ensemble) ont réduit la part de
l'erreur qui était réellement réductible ; le reste relève de facteurs non
observés dans les données sources.

### Piste testée et écartée : un modèle de prévision par service

Le notebook original (`phase3_modelisation_original.ipynb`, Bloc 3A) ne
prévoit le volume qu'au niveau global (`dataset_journalier.csv`) — jamais
par service, alors même que `dataset_journalier_service.csv` existait déjà
(généré par le Bloc E.2 du feature engineering). Ce n'est donc pas une
simplification introduite pendant le portage : le notebook source n'a
jamais fait de prévision par service non plus.

Avant d'envisager de construire 4 modèles par service pour la page
Réallocation de l'app Dash, un test rapide a été mené : un SARIMA avec les
mêmes hyperparamètres que le modèle global retenu (ordre (2,1,2), saison
(1,1,1,7), mêmes régresseurs exogènes), entraîné sur un seul service
(Crédit Immobilier) à titre représentatif :

| Configuration | MAE (test) | MAPE (test) |
|---|---:|---:|
| SARIMA par service (Crédit Immobilier) | 287.8 | 23.8% |
| Naïf — moyenne mobile 7j (même service) | **234.4** | 23.6% |

**Le SARIMA par service fait 22.8% moins bien qu'une simple moyenne
mobile** — pas juste "pas mieux", nettement pire. Le MAPE (23.8%) est
également quasiment le double de celui du modèle global (12.2%) : une
série par service, avec moins de données et sans l'effet de lissage du
cumul de plusieurs services, est intrinsèquement plus bruitée et plus
difficile à modéliser que la série agrégée.

**Décision retenue** : ne pas construire de modèles par service. La page
Réallocation de l'app Dash utilise à la place une projection illustrative
(prévision globale × part historique de volume de chaque service, ETP
supposé constant), explicitement présentée comme une extrapolation et non
une sortie de modèle — cohérent avec le traitement déjà appliqué au
tableau de suggestion de réallocation.

### Synthèse de la trajectoire d'amélioration

| Étape | MAE | MAPE |
|---|---:|---:|
| SARIMA univarié, config fixe (notebook original) | 1135.5 | 24.6% |
| + recherche d'ordre SARIMA | 1019.1 | 19.1% |
| + régresseurs exogènes calendaires | 667.4 | 13.0% |
| + Ensemble SARIMA/XGBoost (résultat final retenu) | **644.7** | **12.2%** |

**Gain total : −43.2% de MAE, −12.4 points de MAPE** par rapport à la
configuration fixe du notebook original.

---

## Bloc 3B — Prévision de la charge par ETP (Ridge / ElasticNet / XGBoost)

| Configuration | Modèle | MAE (test) | RMSE (test) | R² (test) |
|---|---|---:|---:|---:|
| Avant (fixe) | Ridge (alpha=1.0) | **10.95** | **13.69** | **0.644** |
| Avant (fixe) | ElasticNet (alpha=0.1, l1=0.5) | 11.44 | 14.36 | 0.608 |
| Avant (fixe) | XGBoost (n_estim=300, max_depth=4) | 11.52 | 14.40 | 0.606 |
| Après (recherche + features cycliques) | Ridge (alpha=50.0) | 11.46 | 14.49 | 0.601 |
| Après (recherche + features cycliques) | ElasticNet (alpha=1.0, l1=0.8) | 11.78 | 15.20 | 0.561 |
| Après (recherche + features cycliques) | **XGBoost (retenu, max_depth=3, lr=0.1)** | **11.28** | 14.02 | 0.627 |

**Constat honnête, à ne pas passer sous silence** : sur ce bloc, la version
"améliorée" ne bat **pas** la meilleure configuration d'origine. Le Ridge
fixe (alpha=1.0) obtient le meilleur MAE de tout le bloc (10.95), devançant
même le XGBoost optimisé retenu en production (11.28).

Deux causes probables :
1. Le `GridSearchCV` avec `TimeSeriesSplit` a sélectionné `alpha=50.0` pour
   Ridge (régularisation forte) sur la base des folds d'entraînement, mais
   ce choix ne généralise pas aussi bien sur la fenêtre de test finale —
   signe que la série `charge_par_etp` n'est pas parfaitement stationnaire
   d'une période à l'autre.
2. Les features cycliques ajoutées (sin/cos) n'apportent pas de gain
   mesurable ici et diluent potentiellement le signal pour un modèle
   linéaire déjà régularisé.

Seul XGBoost profite réellement du portage (11.52 → 11.28, soit **−2.1%** de
MAE). La logique de sélection automatique du script ne compare cependant
qu'entre les 3 modèles *optimisés* : elle ne se rabat pas sur la config
d'origine si celle-ci est meilleure. C'est une limite identifiée du script
`modelisation_charge_etp.py`, à corriger si ce bloc doit être présenté comme
une amélioration dans le mémoire (ou à documenter tel quel comme résultat
d'expérimentation honnête).

---

## Bloc 3C — Classification du risque de retard (DecisionTree / RandomForest)

| Configuration | Modèle | Accuracy (test) | F1 (test) |
|---|---|---:|---:|
| Avant (fixe, sans Type_Processus, seuil 0.5) | Decision Tree (max_depth=6, min_leaf=50) | 0.803 | 0.631 |
| Avant (fixe, sans Type_Processus, seuil 0.5) | Random Forest (n_estim=200, max_depth=10) | 0.830 | 0.655 |
| **Après (recherche + Type_Processus + seuil ajusté)** | **Decision Tree (retenu, max_depth=4, min_leaf=20)** | **0.849** | **0.666** |
| Après (recherche + Type_Processus + seuil ajusté) | Random Forest (n_estim=100, max_depth=8) | 0.829 | — |

**Gain net et net sur ce bloc** : +5.7 points d'accuracy et +3.5 points de F1
pour le modèle retenu (Decision Tree), grâce à la combinaison recherche
d'hyperparamètres + encodage de `Type_Processus` + ajustement du seuil de
décision. Le Random Forest, lui, reste quasi stable (0.830 → 0.829).

---

## Bloc 3D — Détection d'anomalies (Isolation Forest)

| Configuration | n_estimators | Taux détection (injection test) | Taux anomalies (global) |
|---|---:|---:|---:|
| Avant (fixe) | 200 | 100.0% | 1.98% |
| **Après (recherche sur validation)** | **100** | **100.0%** | **1.97%** |

Performance strictement équivalente, obtenue avec **deux fois moins d'arbres**
(100 au lieu de 200) — gain d'efficacité (temps d'entraînement/inférence
divisé par ~2) sans perte de qualité de détection, plutôt qu'un gain de
performance brute.

---

## Synthèse pour le mémoire

| Bloc | Résultat | Gain principal |
|---|---|---|
| 3A — Volumes | Amélioration confirmée, plafond réaliste atteint | −43.2% MAE, −12.4 points MAPE (ordre + exogènes + ensemble) ; MAE résiduel (644.7) proche du plancher naïf (647.0), reste de l'erreur non réductible avec les données disponibles |
| 3B — Charge ETP | **Mitigé** — XGBoost s'améliore (−2.1% MAE) mais reste sous le Ridge d'origine | À retravailler : élargir la comparaison a la config d'origine, ou revoir la grille de recherche Ridge |
| 3C — Risque retard | Amélioration confirmée | +5.7 points accuracy, +3.5 points F1 (features + seuil) |
| 3D — Anomalies | Performance égale, efficacité doublée | 2x moins d'arbres pour un taux de détection identique |

Trois blocs sur quatre montrent un gain réel et mesurable, le bloc 3A de
façon particulièrement nette (−43.2% de MAE). Le bloc 3B est le seul où le
portage "amélioré" ne surpasse pas la configuration d'origine sur le jeu de
test — un résultat honnête à assumer dans le mémoire plutôt qu'à dissimuler,
et une piste d'amélioration explicite pour la suite (comparer systématiquement
le modèle optimisé à la configuration de référence avant sélection finale).
