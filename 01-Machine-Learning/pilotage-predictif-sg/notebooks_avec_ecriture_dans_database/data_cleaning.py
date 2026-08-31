"""
data_cleaning.py

Port fidele du notebook data_cleaning.ipynb (9 blocs), adapte pour lire
depuis les tables STAGING et ecrire dans les tables CLEAN de PostgreSQL
au lieu de fichiers CSV.

Regle : ce script ne fait qu'une chose (nettoyer) et ECRIT son resultat
dans la base. Aucune variable transmise a un autre script en memoire.

Prerequis : setup_database.py deja execute (tables clean existantes),
import_staging.py deja execute (staging peuple).

Usage :
    python data_cleaning.py
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))
from db_connection import get_engine

warnings.filterwarnings("ignore")

engine = get_engine()

# Renommage staging (snake_case, issu de Postgres) -> CamelCase
# (nommage utilise dans le reste du pipeline : feature engineering, Dash)
RENAME_TACHES = {
    "id_tache": "ID_Tache",
    "matricule_agent": "Matricule_Agent",
    "type_contrat": "Type_Contrat",
    "service": "Service",
    "type_processus": "Type_Processus",
    "date_creation": "Date_Creation",
    "date_prise_en_charge": "Date_Prise_En_Charge",
    "date_cloture": "Date_Cloture",
    "temps_passe_declare_min": "Temps_Passe_Declare_Min",
    "volume_dossiers": "Volume_Dossiers",
    "statut": "Statut",
    "complexite": "Complexite",
    "commentaire": "Commentaire",
    "source_fichier": "source_fichier",
}

FORMATS_DATE_CONNUS = ["%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%y"]


def parser_date(serie):
    """
    Parse une serie de dates contenant plusieurs formats melanges.
    Essaie successivement chaque format connu (vectorise, donc rapide
    meme sur de gros volumes), contrairement a pd.to_datetime() seul
    qui infere un seul format pour toute la colonne et genere massivement
    des NaT sur des colonnes a formats heterogenes.
    """
    result = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")
    remaining = serie.notna()
    for fmt in FORMATS_DATE_CONNUS:
        if not remaining.any():
            break
        parsed = pd.to_datetime(serie[remaining], format=fmt, errors="coerce")
        ok = parsed.notna()
        idx = remaining[remaining].index[ok.values]
        result.loc[idx] = parsed[ok].values
        remaining.loc[idx] = False
    return result


def run():
    print("=" * 60)
    print("PHASE 1 - DATA CLEANING (PostgreSQL)")
    print("=" * 60)

    # --- BLOC 1 : Chargement depuis staging (remplace le chargement fichiers) ---
    print("\nBLOC 1 : Chargement depuis les tables staging...")
    df_2024 = pd.read_sql("SELECT * FROM staging_taches_2024", engine)
    df_2025 = pd.read_sql("SELECT * FROM staging_taches_2025", engine)
    df_agents = pd.read_sql("SELECT * FROM staging_agents", engine)
    df_absences = pd.read_sql("SELECT * FROM staging_absences", engine)

    df_2024 = df_2024.rename(columns=RENAME_TACHES)
    df_2025 = df_2025.rename(columns=RENAME_TACHES)
    df_absences = df_absences.rename(columns={
        "matricule": "Matricule", "type_contrat": "Type_Contrat", "service": "Service",
        "type_absence": "Type_Absence", "date_debut": "Date_Debut", "date_fin": "Date_Fin",
        "duree_jours": "Duree_Jours", "valide": "Valide", "commentaire_rh": "Commentaire_RH",
    })
    print(f"  staging_taches_2024 : {len(df_2024)} lignes")
    print(f"  staging_taches_2025 : {len(df_2025)} lignes")
    print(f"  staging_agents      : {len(df_agents)} lignes")
    print(f"  staging_absences    : {len(df_absences)} lignes")

    # --- BLOC 2 : Fusion 2024 + 2025 ---
    print("\nBLOC 2 : Fusion des taches 2024 + 2025...")
    df_taches = pd.concat([df_2024, df_2025], ignore_index=True)
    print(f"  Fusion 2024 + 2025 : {len(df_taches)} lignes")

    # --- BLOC 3 : Doublons ---
    print("\nBLOC 3 : Suppression des doublons...")
    avant = len(df_taches)
    df_taches = df_taches.drop_duplicates(subset=["ID_Tache"], keep="first")
    print(f"  Taches   : {avant - len(df_taches)} doublons supprimes -> {len(df_taches)} lignes")

    avant_abs = len(df_absences)
    df_absences = df_absences.drop_duplicates(keep="first")
    print(f"  Absences : {avant_abs - len(df_absences)} doublons supprimes -> {len(df_absences)} lignes")

    avant_ag = len(df_agents)
    df_agents = df_agents.drop_duplicates(subset=["matricule"], keep="first")
    print(f"  Agents   : {avant_ag - len(df_agents)} doublons supprimes -> {len(df_agents)} lignes")

    # --- BLOC 4 : Valeurs manquantes ---
    print("\nBLOC 4 : Valeurs manquantes...")
    df_taches = df_taches.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    df_agents = df_agents.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    df_absences = df_absences.replace({"": np.nan, "None": np.nan, "nan": np.nan})

    avant = len(df_taches)
    df_taches = df_taches.dropna(subset=["ID_Tache"])
    print(f"  Lignes sans ID_Tache supprimees : {avant - len(df_taches)}")

    avant = len(df_taches)
    df_taches = df_taches.dropna(subset=["Matricule_Agent"])
    print(f"  Lignes sans Matricule supprimees : {avant - len(df_taches)}")

    # --- BLOC 5 : Harmonisation des dates ---
    print("\nBLOC 5 : Harmonisation des dates...")
    for col in ["Date_Creation", "Date_Prise_En_Charge", "Date_Cloture"]:
        df_taches[col] = parser_date(df_taches[col])
        print(f"  {col} parsee")

    for col in ["date_entree", "date_fin_contrat"]:
        if col in df_agents.columns:
            df_agents[col] = parser_date(df_agents[col])

    for col in ["Date_Debut", "Date_Fin"]:
        if col in df_absences.columns:
            df_absences[col] = parser_date(df_absences[col])
    print("  Toutes les dates sont maintenant au format datetime")

    # --- BLOC 6 : Normalisation des categories ---
    print("\nBLOC 6 : Normalisation des categories...")
    mapping_statuts = {
        "CLOTURE": "Clôturé", "cloture": "Clôturé", "Clôturé": "Clôturé",
        "Fermé": "Clôturé", "EN_COURS": "En cours", "En cours": "En cours",
        "Suspendu": "Suspendu",
    }
    df_taches["Statut"] = df_taches["Statut"].str.strip().map(mapping_statuts).fillna("Inconnu")

    mapping_complexite = {
        "Simple": "Simple", "simple": "Simple",
        "Moyen": "Moyen", "moyen": "Moyen",
        "Complexe": "Complexe", "COMPLEXE": "Complexe",
    }
    df_taches["Complexite"] = df_taches["Complexite"].str.strip().map(mapping_complexite).fillna("Non renseigné")

    mapping_contrat = {
        "CDI": "CDI", "cdi": "CDI",
        "CDD": "CDD", "cdd": "CDD",
        "Intérimaire": "Intérimaire", "Interimaire": "Intérimaire", "INTERIMAIRE": "Intérimaire",
        "Alternant": "Alternant", "alternant": "Alternant",
        "Prestataire": "Prestataire", "PRESTATAIRE": "Prestataire",
        "Temps partiel": "Temps partiel",
    }
    df_taches["Type_Contrat"] = df_taches["Type_Contrat"].str.strip().map(mapping_contrat).fillna("Non renseigné")
    df_agents["type_contrat"] = df_agents["type_contrat"].str.strip().map(mapping_contrat).fillna("Non renseigné")
    df_absences["Type_Contrat"] = df_absences["Type_Contrat"].str.strip().map(mapping_contrat).fillna("Non renseigné")

    mapping_absences = {
        "CP": "Congé Payé", "Congé Payé": "Congé Payé", "CONGE_PAYE": "Congé Payé",
        "Maladie": "Maladie", "MALADIE": "Maladie",
        "RTT": "RTT", "rtt": "RTT",
        "Formation": "Formation", "FORMATION": "Formation",
        "Absent": "Absence non justifiée",
        "Semaine école": "Semaine école", "SEMAINE_ECOLE": "Semaine école",
        "Formation CFA": "Semaine école",
    }
    df_absences["Type_Absence"] = df_absences["Type_Absence"].str.strip().map(mapping_absences).fillna("Autre")
    print("  Statuts, complexite, contrats, absences normalises")

    # --- BLOC 7 : Valeurs aberrantes ---
    print("\nBLOC 7 : Valeurs aberrantes...")
    df_taches["Temps_Passe_Declare_Min"] = pd.to_numeric(df_taches["Temps_Passe_Declare_Min"], errors="coerce")
    df_taches["Volume_Dossiers"] = pd.to_numeric(df_taches["Volume_Dossiers"], errors="coerce")
    df_absences["Duree_Jours"] = pd.to_numeric(df_absences["Duree_Jours"], errors="coerce")

    nb_temps_neg = (df_taches["Temps_Passe_Declare_Min"] < 0).sum()
    df_taches.loc[df_taches["Temps_Passe_Declare_Min"] < 0, "Temps_Passe_Declare_Min"] = np.nan
    print(f"  Temps negatifs mis a NaN       : {nb_temps_neg} valeurs")

    nb_vol_aber = (df_taches["Volume_Dossiers"] > 500).sum()
    df_taches.loc[df_taches["Volume_Dossiers"] > 500, "Volume_Dossiers"] = np.nan
    print(f"  Volumes aberrants mis a NaN    : {nb_vol_aber} valeurs")

    nb_abs_aber = ((df_absences["Duree_Jours"] < 0) | (df_absences["Duree_Jours"] > 90)).sum()
    df_absences.loc[(df_absences["Duree_Jours"] < 0) | (df_absences["Duree_Jours"] > 90), "Duree_Jours"] = np.nan
    print(f"  Durees absences aberrantes NaN : {nb_abs_aber} valeurs")

    nb_incoherents = (df_taches["Date_Cloture"] < df_taches["Date_Creation"]).sum()
    df_taches.loc[df_taches["Date_Cloture"] < df_taches["Date_Creation"], "Date_Cloture"] = np.nan
    print(f"  Dates incoherentes (cloture < creation) : {nb_incoherents} corrigees")

    # --- BLOC 8 : Jointure agents <-> taches ---
    print("\nBLOC 8 : Jointure des datasets...")
    df_agents_renamed = df_agents.rename(columns={
        "matricule": "Matricule_Agent",
        "service": "Service_Agent",
        "type_contrat": "Type_Contrat_Agent",
        "temps_travail": "Temps_Travail",
        "date_entree": "Date_Entree",
        "date_fin_contrat": "Date_Fin_Contrat",
        "actif": "Actif",
    })

    df_final = df_taches.merge(
        df_agents_renamed[["Matricule_Agent", "Service_Agent", "Type_Contrat_Agent",
                            "Temps_Travail", "Date_Entree", "Date_Fin_Contrat", "Actif"]],
        on="Matricule_Agent", how="left",
    )
    nb_fantomes = df_final["Service_Agent"].isna().sum()
    print(f"  Taches sans agent reference (fantomes) : {nb_fantomes}")
    print(f"  Dataset final apres jointure : {len(df_final)} lignes | {df_final.shape[1]} colonnes")

    # --- BLOC 9 : Ecriture dans les tables CLEAN (remplace l'export CSV) ---
    print("\nBLOC 9 : Ecriture dans PostgreSQL...")

    # taches_clean : colonnes renommees en snake_case pour coller au schema SQL
    df_taches_clean_sql = df_final.rename(columns={
        "ID_Tache": "id_tache", "Matricule_Agent": "matricule_agent",
        "Type_Contrat": "type_contrat", "Service": "service",
        "Type_Processus": "type_processus", "Date_Creation": "date_creation",
        "Date_Prise_En_Charge": "date_prise_en_charge", "Date_Cloture": "date_cloture",
        "Temps_Passe_Declare_Min": "temps_passe_declare_min", "Volume_Dossiers": "volume_dossiers",
        "Statut": "statut", "Complexite": "complexite", "Commentaire": "commentaire",
        "Service_Agent": "service_agent", "Type_Contrat_Agent": "type_contrat_agent",
        "Temps_Travail": "temps_travail", "Date_Entree": "date_entree",
        "Date_Fin_Contrat": "date_fin_contrat", "Actif": "actif",
    })
    cols_taches = ["id_tache", "matricule_agent", "type_contrat", "service", "type_processus",
                   "date_creation", "date_prise_en_charge", "date_cloture",
                   "temps_passe_declare_min", "volume_dossiers", "statut", "complexite",
                   "commentaire", "source_fichier", "service_agent", "type_contrat_agent",
                   "temps_travail", "date_entree", "date_fin_contrat", "actif"]
    df_taches_clean_sql = df_taches_clean_sql[cols_taches]

    with engine.begin() as conn:
        conn.exec_driver_sql("TRUNCATE TABLE taches_clean")
    df_taches_clean_sql.to_sql("taches_clean", engine, if_exists="append", index=False, chunksize=5000)
    print(f"  taches_clean    : {len(df_taches_clean_sql)} lignes ecrites")

    df_agents.to_sql("agents_clean", engine, if_exists="replace", index=False)
    print(f"  agents_clean    : {len(df_agents)} lignes ecrites")

    df_absences_sql = df_absences.rename(columns={
        "Matricule": "matricule", "Type_Contrat": "type_contrat", "Service": "service",
        "Type_Absence": "type_absence", "Date_Debut": "date_debut", "Date_Fin": "date_fin",
        "Duree_Jours": "duree_jours", "Valide": "valide", "Commentaire_RH": "commentaire_rh",
    })
    df_absences_sql.to_sql("absences_clean", engine, if_exists="replace", index=False)
    print(f"  absences_clean  : {len(df_absences_sql)} lignes ecrites")

    # --- RAPPORT FINAL ---
    print("\n" + "=" * 60)
    print("RAPPORT FINAL - DONNEES PROPRES")
    print("=" * 60)
    print(f"  Lignes totales    : {len(df_final)}")
    print(f"  Periode couverte  : {df_final['Date_Creation'].min().date()} -> {df_final['Date_Creation'].max().date()}")
    print(f"  Agents uniques    : {df_final['Matricule_Agent'].nunique()}")
    print(f"  Services uniques  : {df_final['Service'].nunique()}")
    print("\nPhase 1 terminee. Pret pour le Feature Engineering.")


if __name__ == "__main__":
    run()
