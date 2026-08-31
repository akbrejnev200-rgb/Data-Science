"""
run_pipeline.py

Orchestrateur : execute toutes les etapes du pipeline dans l'ordre, en
appelant les fonctions de chaque module (aucun code duplique). C'est
l'equivalent "fait maison" de ce que ferait un outil comme Airflow pour
un pipeline de cette taille.

Chaque etape lit son entree depuis PostgreSQL et ecrit son resultat
dans PostgreSQL avant de passer a la suivante : jamais de variable
Python transmise directement entre etapes.

Deux modes pour la Phase 3 (modelisation) :
- Mode complet (par defaut) : reentraine les 4 modeles (couteux --
  recherche d'hyperparametres, comparaison de candidats), PUIS genere des
  predictions fraiches avec les modeles tout juste reentraines.
- --predict-only : ne reentraine rien, se contente d'appeler les 4
  scripts predict_*.py sur les modeles .pkl deja sauvegardes -- c'est ce
  mode qui est fait pour tourner chaque jour (rafraichissement des
  previsions/backtests avec les dernieres donnees), le reentrainement
  complet restant occasionnel (hebdomadaire/mensuel).

Usage :
    python run_pipeline.py                      # toutes les etapes (fit + predict)
    python run_pipeline.py --skip-import         # sans re-importer les CSV bruts
    python run_pipeline.py --skip-modelisation   # s'arreter apres les features (Phase 1+2 seulement)
    python run_pipeline.py --predict-only        # rafraichissement quotidien : ingestion + features + predict seul
"""

import argparse
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))
import setup_database
import import_staging
from db_connection import get_engine

import data_cleaning
import feature_engineering
import modelisation_volumes
import modelisation_charge_etp
import modelisation_retard
import modelisation_anomalies
import predict_volumes
import predict_charge
import predict_retard
import predict_anomalies


def _predict_tout():
    print("\n" + "#" * 60)
    print("# Bloc 3A : predict volumes")
    print("#" * 60)
    predict_volumes.run()

    print("\n" + "#" * 60)
    print("# Bloc 3B : predict charge par ETP")
    print("#" * 60)
    predict_charge.run()

    print("\n" + "#" * 60)
    print("# Bloc 3C : predict risque de retard")
    print("#" * 60)
    predict_retard.run()

    print("\n" + "#" * 60)
    print("# Bloc 3D : predict anomalies")
    print("#" * 60)
    predict_anomalies.run()


def main(skip_import: bool = False, skip_modelisation: bool = False, predict_only: bool = False):
    engine = get_engine()

    print("\n" + "#" * 60)
    print("# ETAPE 1 - Creation des tables (staging + clean)")
    print("#" * 60)
    with engine.begin() as conn:
        conn.execute(text(setup_database.CREATE_TABLES_SQL))
    print("Tables staging + clean creees (ou deja existantes).")

    if not skip_import:
        print("\n" + "#" * 60)
        print("# ETAPE 2 - Import des fichiers bruts vers staging")
        print("#" * 60)
        import_staging.import_taches_2024()
        import_staging.import_taches_2025()
        import_staging.import_agents()
        import_staging.import_absences()
    else:
        print("\n(Import staging ignore : --skip-import)")

    print("\n" + "#" * 60)
    print("# ETAPE 3 - Nettoyage (staging -> clean)")
    print("#" * 60)
    data_cleaning.run()

    print("\n" + "#" * 60)
    print("# ETAPE 4 - Feature engineering (clean -> features)")
    print("#" * 60)
    feature_engineering.run()

    if predict_only:
        print("\n" + "#" * 60)
        print("# ETAPE 5 - Predict seul (rafraichissement quotidien, sans reentrainement)")
        print("#" * 60)
        _predict_tout()
    elif skip_modelisation:
        print("\n(Modelisation ignoree : --skip-modelisation)")
    else:
        print("\n" + "#" * 60)
        print("# ETAPE 5 - Fit : Bloc 3A prevision des volumes")
        print("#" * 60)
        modelisation_volumes.run()

        print("\n" + "#" * 60)
        print("# ETAPE 6 - Fit : Bloc 3B prevision de la charge par ETP")
        print("#" * 60)
        modelisation_charge_etp.run()

        print("\n" + "#" * 60)
        print("# ETAPE 7 - Fit : Bloc 3C classification du risque de retard")
        print("#" * 60)
        modelisation_retard.run()

        print("\n" + "#" * 60)
        print("# ETAPE 8 - Fit : Bloc 3D detection d'anomalies")
        print("#" * 60)
        modelisation_anomalies.run()

        print("\n" + "#" * 60)
        print("# ETAPE 9 - Predict (avec les modeles tout juste reentraines)")
        print("#" * 60)
        _predict_tout()

    print("\n" + "#" * 60)
    print("# PIPELINE TERMINE")
    print("#" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-import", action="store_true",
        help="Ne pas re-importer les fichiers bruts (staging deja peuple)",
    )
    parser.add_argument(
        "--skip-modelisation", action="store_true",
        help="S'arreter apres feature engineering, sans lancer la Phase 3 (ignore si --predict-only)",
    )
    parser.add_argument(
        "--predict-only", action="store_true",
        help="Rafraichissement quotidien : ingestion + features + predict seul (aucun reentrainement)",
    )
    args = parser.parse_args()
    main(skip_import=args.skip_import, skip_modelisation=args.skip_modelisation, predict_only=args.predict_only)
