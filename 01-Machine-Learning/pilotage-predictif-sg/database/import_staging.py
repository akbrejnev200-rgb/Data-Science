"""
import_staging.py

Importe les 4 fichiers SOURCES BRUTS (et uniquement ceux-la) dans les
tables staging (couche bronze).

IMPORTANT : seuls ces 4 fichiers sont des sources brutes. Tout autre
fichier CSV/PKL du projet (dataset_clean.csv, dataset_features.csv,
dataset_journalier*.csv, absences_clean.csv, previsions_volumes_j30.csv,
anomalies_detectees.csv, model_*.pkl) est une SORTIE generee par le code
(data_cleaning.py, feature_engineering.py, modelisation) et ne doit jamais
etre importe ici. Ces sorties doivent etre regenerees en executant le
pipeline contre la base, pas recuperees telles quelles.

Usage :
    python import_staging.py
"""

import pandas as pd
import json
from db_connection import get_engine, DATA_DIR

engine = get_engine()


def import_taches_2024():
    df = pd.read_csv(DATA_DIR / "extractions_taches_2024.csv", dtype=str, encoding="utf-8")
    df["source_fichier"] = "extractions_taches_2024.csv"
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.to_sql("staging_taches_2024", engine, if_exists="append", index=False)
    print(f"staging_taches_2024 : {len(df)} lignes importees")


def import_taches_2025():
    df = pd.read_excel(DATA_DIR / "extractions_taches_2025.xlsx", sheet_name="Taches_2025", dtype=str)
    df = df.dropna(how="all")
    df["source_fichier"] = "extractions_taches_2025.xlsx"

    rename_map = {
        "id tache": "id_tache",
        "MATRICULE AGENT": "matricule_agent",
        "type contrat": "type_contrat",
        "Service ": "service",
        "type_processus": "type_processus",
        "date creation": "date_creation",
        "Date Prise En Charge": "date_prise_en_charge",
        "DATE_CLOTURE": "date_cloture",
        "tps_passe (min)": "temps_passe_declare_min",
        "Nb Dossiers": "volume_dossiers",
        "STATUT": "statut",
        "complexite": "complexite",
        "commentaires": "commentaire",
    }
    df = df.rename(columns=rename_map)
    df.to_sql("staging_taches_2025", engine, if_exists="append", index=False)
    print(f"staging_taches_2025 : {len(df)} lignes importees")


def import_agents():
    with open(DATA_DIR / "agents_equipes.json", "r", encoding="utf-8") as f:
        agents = json.load(f)
    df = pd.DataFrame(agents).astype(str).replace("nan", None)
    df.to_sql("staging_agents", engine, if_exists="append", index=False)
    print(f"staging_agents : {len(df)} entrees importees")


def import_absences():
    df = pd.read_csv(DATA_DIR / "absences_conges.csv", dtype=str, encoding="utf-8")
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.to_sql("staging_absences", engine, if_exists="append", index=False)
    print(f"staging_absences : {len(df)} lignes importees")


if __name__ == "__main__":
    import_taches_2024()
    import_taches_2025()
    import_agents()
    import_absences()
    print("Import termine.")
