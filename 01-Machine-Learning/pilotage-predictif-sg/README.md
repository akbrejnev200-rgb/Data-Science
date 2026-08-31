# README — Migration du pipeline CSV vers PostgreSQL
## Projet : Pilotage predictif de la performance operationnelle (memoire Master)

---

## REGLE LA PLUS IMPORTANTE DE CE PROJET

Le projet existait deja, entierement fonctionnel, base sur des fichiers
CSV/XLSX/JSON et des notebooks Colab. On a migre CE MEME pipeline vers
PostgreSQL. Il y a deux categories de fichiers radicalement differentes,
a ne JAMAIS confondre :

### Categorie A — SOURCES BRUTES (les seules importees dans la base)
Ce sont les 4 fichiers d'origine, jamais generes par du code Python :
- `data/raw/extractions_taches_2024.csv`
- `data/raw/extractions_taches_2025.xlsx`
- `data/raw/agents_equipes.json`
- `data/raw/absences_conges.csv`

Ce sont les SEULS fichiers lus avec pandas et inseres dans les tables
`staging_*`, dans `database/import_staging.py`.

### Categorie B — SORTIES GENEREES PAR LE CODE (jamais reimportees)
Les fichiers CSV/PKL de l'ancien pipeline Colab (dataset_clean.csv,
dataset_features.csv, previsions_volumes_j30.csv, etc.) sont des RESULTATS,
pas des sources. Ils ne doivent jamais etre relus/importes directement dans
la base — tout repart des 4 sources brutes et traverse
staging -> clean -> features -> predictions, entierement recalcule depuis
PostgreSQL.

---

## Etat du projet

Base PostgreSQL `pilotage_operationnel` (Windows, pgAdmin), pipeline
entierement porte et teste contre la base reelle :

```
STAGING (bronze)      : staging_taches_2024, staging_taches_2025,
                         staging_agents, staging_absences
CLEAN (silver)        : taches_clean, agents_clean, absences_clean
FEATURES (gold)       : features_detail, features_journalier,
                         features_journalier_service
PREDICTIONS/RESULTATS : predictions_volumes_j30, predictions_charge_etp,
                         predictions_risque_retard, anomalies_detectees
```

### Connexion et alimentation brute (`database/`)

Ne contient que ce qui concerne directement la base : la connexion partagee
et l'import des sources brutes.

- `db_connection.py` : fonction `get_engine()` partagee + `DATA_DIR` / `MODELS_DIR`
- `setup_database.py` : CREATE TABLE staging + clean
- `import_staging.py` : import des 4 sources brutes (categorie A)

### Transformation et modelisation (`notebooks_avec_ecriture_dans_database/`)

Tout ce qui LIT et ECRIT dans la base au fil du pipeline (nettoyage,
enrichissement, modelisation, orchestration), plus le notebook de
verification. Chaque script importe `db_connection` (et `setup_database` /
`import_staging` pour `run_pipeline.py`) depuis `../database` via un ajout
explicite au `sys.path` en tete de fichier — la logique de connexion reste
centralisee dans `database/db_connection.py`, jamais dupliquee.

- `data_cleaning.py` : Phase 1 — nettoyage (9 blocs, port du notebook original)
- `feature_engineering.py` : Phase 2 — enrichissement (blocs A-E)
- `modelisation_common.py` : utilitaire partage (split temporel 70/15/15)
- `modelisation_volumes.py` : Bloc 3A — prevision des volumes (Prophet/SARIMA)
- `modelisation_charge_etp.py` : Bloc 3B — charge par ETP (Ridge/ElasticNet/XGBoost)
- `modelisation_retard.py` : Bloc 3C — classification du risque de retard
- `modelisation_anomalies.py` : Bloc 3D — detection d'anomalies (Isolation Forest + SHAP)
- `run_pipeline.py` : orchestrateur (toutes les etapes dans l'ordre)
- `verification.ipynb` : notebook d'exploration/validation en lecture seule
  (aucune ecriture en base). Importe `db_connection` via `Path.cwd().parent
  / "database"` ajoute au `sys.path` en premiere cellule.

### Point d'attention connu : Prophet indisponible sous Windows

Le binaire Stan precompile de Prophet plante systematiquement sur cet
environnement (probleme d'environnement Windows documente, independant du
code — VC++ redistributable, recompilation cmdstan depuis les sources,
exclusions antivirus : aucune de ces pistes n'a resolu le probleme).
`modelisation_volumes.py` gere ce cas avec un repli automatique sur
SARIMA/XGBoost (SARIMA desormais enrichi de regresseurs exogenes
calendaires, ce qui a fait chuter le MAE test de 1019 a 667, soit -34.5%),
documente dans le rapport final du script et dans `models/model_volumes.pkl`.

Pour evaluer Prophet malgre tout, voir `colab_prophet/` : notebook
autonome a executer sur Google Colab (Linux, ou Prophet fonctionne), avec
script de reintegration automatique du resultat si celui-ci bat le modele
local (`colab_prophet/README.md` pour la marche a suivre complete).

## Documentation

- `docs/AMELIORATION_MODELES.md` : comparaison chiffree avant/apres entre
  les hyperparametres fixes des notebooks originaux et les versions
  optimisees (recherche d'hyperparametres, nouvelles features, ajustement
  de seuil), bloc par bloc.

## Notebooks originaux (`notebooks_originaux/`)

Conserves comme reference (methodologie source, hyperparametres d'origine
utilises pour la comparaison avant/apres) :
- `data_cleaning_original.ipynb`
- `feature_engineering_original.ipynb`
- `phase3_modelisation_original.ipynb`

`archive/pipeline_complet_original.ipynb` : simple concatenation des 3
notebooks ci-dessus pour execution Colab en une fois, sans logique propre —
archive car redondant maintenant que les 3 notebooks sources sont portes et
testes individuellement.

## Environnement technique

- Windows, VS Code, venv Python dans `C:\Users\akbre\Downloads\python\venv`
  (au niveau AU-DESSUS du dossier `mon_projet`)
- PostgreSQL local, base `pilotage_operationnel`, identifiants dans `.env`
  a la racine du projet (non commite sur Git)

### Installation

**Prerequis** : Python 3.10+, PostgreSQL 14+ (teste avec PostgreSQL 18).

**1. Dependances Python**

Listees dans `requirements.txt` (genere depuis le venv du projet via
`pip freeze`) :

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Base de donnees**

Creer la base puis restaurer le dump fourni (`database/dump_pilotage_operationnel.sql.gz`,
qui contient le schema et les donnees des couches gold/predictions utilisees
par l'app -- `features_*`, `predictions_*`, `anomalies_detectees`,
`incidents_manager`, `agents_clean`, `absences_clean`. Les tables bronze/silver
intermediaires -- `staging_taches_2024/2025`, `taches_clean` -- ne sont pas
incluses car non lues par l'app ; elles peuvent etre regenerees en rejouant
le pipeline complet depuis `data/raw/` via
`notebooks_avec_ecriture_dans_database/run_pipeline.py` si besoin) :

```
createdb -U postgres pilotage_operationnel
gunzip -c database/dump_pilotage_operationnel.sql.gz | psql -U postgres -d pilotage_operationnel
```

**3. Configuration**

Copier `.env.example` vers `.env` a la racine du projet et renseigner les
variables suivantes (voir `database/db_connection.py` et `app/auth.py` pour
leur usage) :

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pilotage_operationnel
DB_USER=postgres
DB_PASSWORD=<mot de passe PostgreSQL local>

FLASK_SECRET_KEY=<cle secrete Flask, ex. genere avec python -c "import secrets; print(secrets.token_hex(32))">
APP_USERNAME=<identifiant de connexion a l'app>
APP_PASSWORD_HASH=<hash genere via app/generer_mot_de_passe.py>
```

**4. Identifiants de connexion**

L'application n'a qu'un seul niveau d'acces (pas de distinction test/admin) :
identifiant et mot de passe definis dans `.env` (`APP_USERNAME` /
`APP_PASSWORD_HASH`), a communiquer separement pour la soutenance/l'evaluation.
Connexion a la base PostgreSQL : voir `DB_USER` / `DB_PASSWORD` ci-dessus.

**5. Lancement**

```
Lancer_application.bat
```
ou directement `python app/app.py`. L'app est servie sur `http://localhost:8050`.

**Version en ligne** : https://pilotage-predictif-sg.onrender.com (deployee sur
Render, connectee a une base PostgreSQL Neon distincte de la base locale --
voir `.env.neon`, non commite). Plan gratuit Render : l'app se met en veille
apres une periode d'inactivite, premier chargement ~30-60s le temps du reveil.

## Contexte du memoire

Memoire de Master Data & IA, alternance Societe Generale, Filiere
Operations Personnes Physiques. Objectif : remplacer le pilotage reactif
de la performance operationnelle par un systeme predictif (app Dash) base
sur SARIMA/Prophet, Ridge/Elastic Net/XGBoost, RandomForest/DecisionTree,
Isolation Forest, SHAP. Deadline memoire : 17 aout 2026. Soutenance :
9 septembre 2026. Contrainte stricte : pipeline 100% PostgreSQL, aucun
export CSV dans la version finale.
