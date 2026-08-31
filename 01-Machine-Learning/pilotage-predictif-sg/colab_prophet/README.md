# Bloc 3A sur Google Colab — Prophet

Prophet plante systematiquement sur l'environnement Windows local (binaire
Stan precompile incompatible, cf. `models/model_volumes.pkl` et le rapport
final de `notebooks_avec_ecriture_dans_database/modelisation_volumes.py`
pour le detail des tentatives de correction). Ce dossier permet d'evaluer
Prophet sur Colab (Linux), ou il fonctionne normalement.

## Contenu

- `prevision_volumes_colab.ipynb` : notebook autonome, reproduit exactement
  la logique du Bloc 3A local (Prophet + SARIMA avec regresseurs exogenes +
  XGBoost, meme split temporel 70/15/15, meme selection par MAE test).
- `features_journalier.csv` : export de la table `features_journalier`
  (a uploader dans Colab).
- `integrer_resultats_colab.py` : script a lancer en local une fois les
  resultats Colab recuperes, pour les integrer au pipeline si (et
  seulement si) ils sont meilleurs que le modele local actuel.
- `resultats/` : dossier ou deposer les fichiers telecharges depuis Colab.

## Marche a suivre

1. Aller sur [colab.research.google.com](https://colab.research.google.com),
   ouvrir `prevision_volumes_colab.ipynb` (Fichier -> Importer un notebook).
2. Menu **Execution -> Tout executer**. Rien a modifier dans le code.
3. A la cellule "Chargement des donnees", un selecteur de fichier s'ouvre :
   uploader `features_journalier.csv` (fourni dans ce dossier).
4. Le notebook entraine et compare Prophet / SARIMA / XGBoost, selectionne
   le meilleur par MAE sur le jeu de test, genere les previsions J+1 a J+30,
   puis telecharge automatiquement 3 fichiers en fin d'execution :
   - `model_volumes.pkl`
   - `predictions_volumes_j30.csv`
   - `metriques_bloc3a.txt`
5. Deposer ces 3 fichiers dans `colab_prophet/resultats/` sur la machine
   locale.
6. Lancer `python integrer_resultats_colab.py` depuis ce dossier
   (PostgreSQL doit etre demarre). Le script compare automatiquement le
   MAE test du modele Colab a celui du modele local :
   - si Colab fait mieux -> `models/model_volumes.pkl` et la table
     `predictions_volumes_j30` sont remplaces par le resultat Colab ;
   - sinon -> rien n'est modifie, le modele local (SARIMA avec exogenes,
     MAE test = 667.4) reste en place.

## Etat de reference (modele local actuel, sans Prophet)

| Modele | MAE test | MAPE test |
|---|---:|---:|
| SARIMA (avec regresseurs exogenes, retenu) | 667.4 | 13.0% |
| XGBoost (features de lag) | 671.0 | 12.5% |

C'est ce score que Prophet doit battre sur Colab pour etre adopte.
