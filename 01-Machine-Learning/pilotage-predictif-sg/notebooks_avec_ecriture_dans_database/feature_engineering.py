"""
feature_engineering.py

Port fidele du notebook feature_engineering.ipynb (blocs A a E), adapte
pour lire depuis les tables CLEAN et ecrire dans les tables FEATURES de
PostgreSQL au lieu de fichiers CSV.

Regle : ce script ne fait qu'une chose (enrichir) et ECRIT son resultat
dans la base. Aucune variable transmise a un autre script en memoire.

Prerequis : data_cleaning.py deja execute (tables clean peuplees).

Usage :
    python feature_engineering.py
"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))
from db_connection import get_engine

warnings.filterwarnings("ignore")

engine = get_engine()


def run():
    print("=" * 60)
    print("PHASE 2 - FEATURE ENGINEERING (PostgreSQL)")
    print("=" * 60)

    # --- Chargement depuis les tables clean (remplace read_csv) ---
    print("\nChargement des tables clean...")
    df = pd.read_sql("SELECT * FROM taches_clean", engine, parse_dates=[
        "date_creation", "date_prise_en_charge", "date_cloture",
        "date_entree", "date_fin_contrat",
    ])
    df_abs = pd.read_sql("SELECT * FROM absences_clean", engine, parse_dates=["date_debut", "date_fin"])

    # Renommage snake_case (SQL) -> CamelCase (logique metier d'origine)
    df = df.rename(columns={
        "id_tache": "ID_Tache", "matricule_agent": "Matricule_Agent",
        "type_contrat": "Type_Contrat", "service": "Service",
        "type_processus": "Type_Processus", "date_creation": "Date_Creation",
        "date_prise_en_charge": "Date_Prise_En_Charge", "date_cloture": "Date_Cloture",
        "temps_passe_declare_min": "Temps_Passe_Declare_Min", "volume_dossiers": "Volume_Dossiers",
        "statut": "Statut", "complexite": "Complexite", "commentaire": "Commentaire",
        "service_agent": "Service_Agent", "type_contrat_agent": "Type_Contrat_Agent",
        "temps_travail": "Temps_Travail", "date_entree": "Date_Entree",
        "date_fin_contrat": "Date_Fin_Contrat", "actif": "Actif",
    })
    df_abs = df_abs.rename(columns={
        "matricule": "Matricule", "type_contrat": "Type_Contrat", "service": "Service",
        "type_absence": "Type_Absence", "date_debut": "Date_Debut", "date_fin": "Date_Fin",
        "duree_jours": "Duree_Jours", "valide": "Valide", "commentaire_rh": "Commentaire_RH",
    })
    print(f"  taches_clean   : {len(df)} lignes | {df.shape[1]} colonnes")
    print(f"  absences_clean : {len(df_abs)} lignes")

    # --- BLOC A : Metriques temporelles ---
    print("\nBLOC A : Metriques temporelles...")
    df["delai_prise_en_charge_j"] = (df["Date_Prise_En_Charge"] - df["Date_Creation"]).dt.days
    df.loc[df["delai_prise_en_charge_j"] < 0, "delai_prise_en_charge_j"] = np.nan
    df.loc[df["delai_prise_en_charge_j"] > 60, "delai_prise_en_charge_j"] = np.nan

    df["duree_traitement_reelle_j"] = (df["Date_Cloture"] - df["Date_Prise_En_Charge"]).dt.days
    df.loc[df["duree_traitement_reelle_j"] <= 0, "duree_traitement_reelle_j"] = np.nan
    df.loc[df["duree_traitement_reelle_j"] > 90, "duree_traitement_reelle_j"] = np.nan

    df["duree_totale_j"] = (df["Date_Cloture"] - df["Date_Creation"]).dt.days
    df.loc[df["duree_totale_j"] <= 0, "duree_totale_j"] = np.nan

    df["jour_semaine"] = df["Date_Creation"].dt.dayofweek
    df["nom_jour"] = df["Date_Creation"].dt.day_name()
    df["mois"] = df["Date_Creation"].dt.month
    df["trimestre"] = df["Date_Creation"].dt.quarter
    df["semaine_annee"] = df["Date_Creation"].dt.isocalendar().week.astype("Int64")
    df["annee"] = df["Date_Creation"].dt.year
    df["semaine_du_mois"] = ((df["Date_Creation"].dt.day - 1) // 7) + 1
    df["est_fin_de_mois"] = (df["Date_Creation"].dt.day >= 25).astype(int)
    df["est_debut_mois"] = (df["Date_Creation"].dt.day <= 5).astype(int)
    df["est_lundi"] = (df["jour_semaine"] == 0).astype(int)
    df["est_vendredi"] = (df["jour_semaine"] == 4).astype(int)
    derniers_mois_trimestre = [3, 6, 9, 12]
    df["est_fin_trimestre"] = df["mois"].isin(derniers_mois_trimestre).astype(int)
    print("  Variables temporelles creees")

    # --- BLOC B : Performance par agent ---
    print("\nBLOC B : Performance par agent...")
    df["productivite_dossiers_par_heure"] = np.where(
        (df["Temps_Passe_Declare_Min"] > 0) & (df["Volume_Dossiers"] > 0),
        df["Volume_Dossiers"] / (df["Temps_Passe_Declare_Min"] / 60), np.nan,
    )
    df.loc[df["productivite_dossiers_par_heure"] > 25, "productivite_dossiers_par_heure"] = np.nan

    charge_jour = (df.groupby(["Matricule_Agent", "Date_Creation"]).size()
                   .reset_index(name="charge_journaliere_nb_taches"))
    df = df.merge(charge_jour, on=["Matricule_Agent", "Date_Creation"], how="left")

    moy_prod_agent = (df.groupby("Matricule_Agent")["productivite_dossiers_par_heure"]
                      .mean().reset_index(name="moy_productivite_agent"))
    df = df.merge(moy_prod_agent, on="Matricule_Agent", how="left")

    df["ecart_temps_declare_vs_reel"] = np.where(
        df["duree_traitement_reelle_j"].notna() & df["Temps_Passe_Declare_Min"].notna(),
        df["Temps_Passe_Declare_Min"] - (df["duree_traitement_reelle_j"] * 8 * 60), np.nan,
    )

    df["anciennete_agent_mois"] = np.where(
        df["Date_Entree"].notna(),
        ((df["Date_Creation"] - df["Date_Entree"]).dt.days / 30).round(1), np.nan,
    )
    df.loc[df["anciennete_agent_mois"] < 0, "anciennete_agent_mois"] = np.nan
    print("  Variables agent creees")

    # --- BLOC C : ETP disponibles par jour ---
    print("\nBLOC C : Calcul des ETP disponibles par jour...")

    def convertir_temps_travail(val):
        if pd.isna(val):
            return 1.0
        val = str(val).strip().replace("%", "")
        try:
            v = float(val)
            return v / 100 if v > 1 else v
        except Exception:
            if "plein" in str(val).lower():
                return 1.0
            if "partiel" in str(val).lower():
                return 0.8
            return 1.0

    df["coeff_temps_travail"] = df["Temps_Travail"].apply(convertir_temps_travail)

    df["contrat_actif_a_date"] = np.where(
        df["Date_Fin_Contrat"].isna(), 1,
        (df["Date_Fin_Contrat"] >= df["Date_Creation"]).astype(int),
    )

    lignes_absence = []
    df_abs_valide = df_abs.dropna(subset=["Date_Debut", "Date_Fin", "Matricule"])
    for _, row in df_abs_valide.iterrows():
        debut, fin = row["Date_Debut"], row["Date_Fin"]
        if fin < debut:
            continue
        nb_jours = (fin - debut).days + 1
        if nb_jours > 90:
            continue
        jours = pd.date_range(debut, fin, freq="D")
        for j in jours:
            lignes_absence.append((row["Matricule"], j))

    df_absence_jours = pd.DataFrame(lignes_absence, columns=["Matricule_Agent", "Date_Creation"])
    df_absence_jours = df_absence_jours.drop_duplicates()
    df_absence_jours["est_absent"] = 1

    df = df.merge(df_absence_jours, on=["Matricule_Agent", "Date_Creation"], how="left")
    df["est_absent"] = df["est_absent"].fillna(0).astype(int)

    df["etp_agent_jour"] = (
        df["coeff_temps_travail"] * df["contrat_actif_a_date"] * (1 - df["est_absent"])
    )

    etp_agent_unique = df.drop_duplicates(subset=["Matricule_Agent", "Date_Creation"])[
        ["Matricule_Agent", "Date_Creation", "Service", "etp_agent_jour"]
    ]

    etp_par_jour_global = (etp_agent_unique.groupby("Date_Creation")["etp_agent_jour"]
                            .sum().reset_index(name="etp_total_disponible_jour"))
    df = df.merge(etp_par_jour_global, on="Date_Creation", how="left")

    etp_par_jour_service = (etp_agent_unique.groupby(["Date_Creation", "Service"])["etp_agent_jour"]
                             .sum().reset_index(name="etp_disponible_service_jour"))
    df = df.merge(etp_par_jour_service, on=["Date_Creation", "Service"], how="left")
    print(f"  ETP moyen/jour (global) : {etp_par_jour_global['etp_total_disponible_jour'].mean():.1f}")

    # --- BLOC D : Variables metier ---
    print("\nBLOC D : Variables metier...")
    moy_temps_service = (df.groupby("Service")["Temps_Passe_Declare_Min"]
                         .mean().reset_index(name="moy_temps_service"))
    df = df.merge(moy_temps_service, on="Service", how="left")
    df["ratio_complexite"] = np.where(
        df["moy_temps_service"] > 0,
        df["Temps_Passe_Declare_Min"] / df["moy_temps_service"], np.nan,
    )

    mapping_complexite_num = {"Simple": 1, "Moyen": 2, "Complexe": 3, "Non renseigné": 2}
    df["complexite_num"] = df["Complexite"].map(mapping_complexite_num).fillna(2)

    mapping_productivite_contrat = {
        "CDI": 1.00, "Prestataire": 0.90, "CDD": 0.85, "Intérimaire": 0.75,
        "Temps partiel": 0.80, "Alternant": 0.60, "Non renseigné": 0.85,
    }
    df["coeff_productivite_contrat"] = df["Type_Contrat"].map(mapping_productivite_contrat).fillna(0.85)

    volume_jour_global = (df.groupby("Date_Creation")["Volume_Dossiers"]
                          .sum().reset_index(name="volume_total_entrant_jour"))
    df = df.merge(volume_jour_global, on="Date_Creation", how="left")

    volume_jour_service = (df.groupby(["Date_Creation", "Service"])["Volume_Dossiers"]
                           .sum().reset_index(name="volume_entrant_service_jour"))
    df = df.merge(volume_jour_service, on=["Date_Creation", "Service"], how="left")

    df["charge_par_etp"] = np.where(
        df["etp_total_disponible_jour"] > 0,
        df["volume_total_entrant_jour"] / df["etp_total_disponible_jour"], np.nan,
    )
    df["charge_par_etp_service"] = np.where(
        df["etp_disponible_service_jour"] > 0,
        df["volume_entrant_service_jour"] / df["etp_disponible_service_jour"], np.nan,
    )

    seuil_alerte = df["charge_par_etp"].quantile(0.75)
    df["alerte_surcharge"] = (df["charge_par_etp"] > seuil_alerte).astype(int)

    seuils_service = df.groupby("Service")["charge_par_etp_service"].quantile(0.75)
    df["seuil_alerte_service"] = df["Service"].map(seuils_service)
    df["alerte_surcharge_service"] = (df["charge_par_etp_service"] > df["seuil_alerte_service"]).astype(int)
    print(f"  Seuil global = {seuil_alerte:.1f} dossiers/ETP")

    # --- BLOC E : Datasets agreges (global + par service) ---
    print("\nBLOC E : Agregation journaliere...")
    df_journalier = df.groupby("Date_Creation").agg(
        nb_taches_jour=("ID_Tache", "count"),
        volume_entrant_jour=("Volume_Dossiers", "sum"),
        etp_disponible=("etp_total_disponible_jour", "first"),
        charge_par_etp=("charge_par_etp", "first"),
        alerte_surcharge=("alerte_surcharge", "first"),
        temps_moyen_traitement=("Temps_Passe_Declare_Min", "mean"),
        duree_traitement_mediane=("duree_traitement_reelle_j", "median"),
        taux_complexes=("complexite_num", "mean"),
        jour_semaine=("jour_semaine", "first"),
        mois=("mois", "first"),
        trimestre=("trimestre", "first"),
        est_fin_de_mois=("est_fin_de_mois", "first"),
        est_fin_trimestre=("est_fin_trimestre", "first"),
        est_lundi=("est_lundi", "first"),
        est_vendredi=("est_vendredi", "first"),
        semaine_du_mois=("semaine_du_mois", "first"),
    ).reset_index()
    df_journalier = df_journalier.rename(columns={"Date_Creation": "date"})
    df_journalier = df_journalier.sort_values("date").reset_index(drop=True)

    for lag in [1, 7, 14, 30]:
        df_journalier[f"volume_lag_{lag}j"] = df_journalier["volume_entrant_jour"].shift(lag)
        df_journalier[f"etp_lag_{lag}j"] = df_journalier["etp_disponible"].shift(lag)

    df_journalier["volume_moy_7j"] = df_journalier["volume_entrant_jour"].rolling(window=7, min_periods=1).mean()
    df_journalier["charge_moy_7j"] = df_journalier["charge_par_etp"].rolling(window=7, min_periods=1).mean()
    print(f"  Dataset journalier (global) : {len(df_journalier)} jours")

    print("\nBLOC E.2 : Agregation journaliere par Service...")
    df_journalier_service = df.groupby(["Date_Creation", "Service"]).agg(
        nb_taches_jour=("ID_Tache", "count"),
        volume_entrant_jour=("Volume_Dossiers", "sum"),
        etp_disponible=("etp_disponible_service_jour", "first"),
        charge_par_etp=("charge_par_etp_service", "first"),
        alerte_surcharge=("alerte_surcharge_service", "first"),
        temps_moyen_traitement=("Temps_Passe_Declare_Min", "mean"),
        duree_traitement_mediane=("duree_traitement_reelle_j", "median"),
        taux_complexes=("complexite_num", "mean"),
        jour_semaine=("jour_semaine", "first"),
        mois=("mois", "first"),
        trimestre=("trimestre", "first"),
        est_fin_de_mois=("est_fin_de_mois", "first"),
        est_fin_trimestre=("est_fin_trimestre", "first"),
        est_lundi=("est_lundi", "first"),
        est_vendredi=("est_vendredi", "first"),
        semaine_du_mois=("semaine_du_mois", "first"),
    ).reset_index()
    df_journalier_service = df_journalier_service.rename(columns={"Date_Creation": "date"})
    df_journalier_service = df_journalier_service.sort_values(["Service", "date"]).reset_index(drop=True)

    for lag in [1, 7, 14, 30]:
        df_journalier_service[f"volume_lag_{lag}j"] = (
            df_journalier_service.groupby("Service")["volume_entrant_jour"].shift(lag)
        )
        df_journalier_service[f"etp_lag_{lag}j"] = (
            df_journalier_service.groupby("Service")["etp_disponible"].shift(lag)
        )
    df_journalier_service["volume_moy_7j"] = (
        df_journalier_service.groupby("Service")["volume_entrant_jour"]
        .rolling(window=7, min_periods=1).mean().reset_index(drop=True)
    )
    df_journalier_service["charge_moy_7j"] = (
        df_journalier_service.groupby("Service")["charge_par_etp"]
        .rolling(window=7, min_periods=1).mean().reset_index(drop=True)
    )
    print(f"  Dataset journalier par service : {len(df_journalier_service)} lignes")

    # --- Ecriture dans les tables FEATURES (remplace l'export CSV) ---
    print("\nEcriture dans PostgreSQL (tables features_*)...")
    # if_exists="replace" : le schema est recree automatiquement a partir
    # des dtypes pandas, car ces tables ont des dizaines de colonnes
    # generees par le code plutot qu'un schema fige a l'avance.
    # SKIP_FEATURES_DETAIL=true (mis par .env.neon) : cette table est trop
    # volumineuse pour le plan gratuit Neon (512 Mo). En local (.env par
    # defaut), elle est ecrite normalement -- les Blocs 3C/3D en local en
    # ont besoin directement, sans passer par le contournement en memoire
    # de predict_3c_3d.py (reserve a Neon).
    if os.getenv("SKIP_FEATURES_DETAIL", "false").lower() == "true":
        print("  features_detail : IGNORE (SKIP_FEATURES_DETAIL=true, plan Neon gratuit)")
    else:
        df.to_sql("features_detail", engine, if_exists="replace", index=False, chunksize=5000)
    print(f"  features_detail             : {len(df)} lignes | {df.shape[1]} colonnes")

    df_journalier.to_sql("features_journalier", engine, if_exists="replace", index=False)
    print(f"  features_journalier         : {len(df_journalier)} jours")

    df_journalier_service.to_sql("features_journalier_service", engine, if_exists="replace", index=False)
    print(f"  features_journalier_service : {len(df_journalier_service)} lignes")

    print("\nPhase 2 terminee. Pret pour la modelisation (Phase 3).")


if __name__ == "__main__":
    run()
