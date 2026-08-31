"""
Script de prediction Blocs 3C et 3D, et export d'une version allegee de
features_detail, sans passer par la table features_detail complete (trop
volumineuse pour le plan gratuit Neon). Calcule les features necessaires en
memoire depuis taches_clean, reserve a un usage DEPLOY_TARGET=neon.

Les 3 exports respectent exactement le schema attendu par app/data/loader.py
(noms de colonnes, casse, semantique) plutot que d'inventer un format
different :
- predictions_risque_retard : agregat de backtest (split, verite terrain,
  prediction) -- pas un score en direct, l'app le lit deja via GROUP BY.
- anomalies_detectees : TOUTES les taches scorees (pas seulement celles
  signalees), colonnes de base capitalisees comme dans features_detail.
- features_detail : version reduite aux seules colonnes utilisees par la
  page Derive (agregation par agent), pas les 57 colonnes completes.
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from database.db_connection import get_engine

# Lecture depuis le local (taches_clean n'existe plus sur Neon, deja
# supprimee car non lue par l'app) ; ecriture vers la cible visee par
# DEPLOY_TARGET (neon en general). Evite un aller-retour de re-upload de
# taches_clean juste pour la regenerer.
_deploy_target = os.environ.pop("DEPLOY_TARGET", None)
read_engine = get_engine()  # local, DEPLOY_TARGET retire de l'environnement
if _deploy_target is not None:
    os.environ["DEPLOY_TARGET"] = _deploy_target
write_engine = get_engine()  # neon si DEPLOY_TARGET etait defini
engine = read_engine  # compat avec le reste du script pour les lectures

FENETRE_TACHES_EN_COURS_JOURS = 30  # doit matcher app/data/loader.py

print("=" * 60)
print("PREDICT - BLOCS 3C et 3D (depuis taches_clean)")
print("=" * 60)

print("\nChargement de taches_clean...")
df = pd.read_sql("SELECT * FROM taches_clean", engine)
print(f"  {len(df)} lignes chargees")

absences = pd.read_sql("SELECT * FROM absences_clean", engine)

print("\nCalcul des features temporelles...")
df["date_creation"] = pd.to_datetime(df["date_creation"], errors="coerce")
df["date_prise_en_charge"] = pd.to_datetime(df["date_prise_en_charge"], errors="coerce")
df["date_cloture"] = pd.to_datetime(df["date_cloture"], errors="coerce")

df["jour_semaine"] = df["date_creation"].dt.dayofweek
df["mois"] = df["date_creation"].dt.month
df["trimestre"] = df["date_creation"].dt.quarter
df["annee"] = df["date_creation"].dt.year
df["semaine_annee"] = df["date_creation"].dt.isocalendar().week.astype(int)
df["semaine_du_mois"] = ((df["date_creation"].dt.day - 1) // 7) + 1
df["est_fin_de_mois"] = (df["date_creation"].dt.day >= 25).astype(int)
df["est_debut_mois"] = (df["date_creation"].dt.day <= 5).astype(int)
df["est_lundi"] = (df["jour_semaine"] == 0).astype(int)
df["est_vendredi"] = (df["jour_semaine"] == 4).astype(int)
df["est_fin_trimestre"] = df["mois"].isin([3, 6, 9, 12]).astype(int)

df["delai_prise_en_charge_j"] = (df["date_prise_en_charge"] - df["date_creation"]).dt.days.clip(0, 60)
df["duree_traitement_reelle_j"] = (df["date_cloture"] - df["date_prise_en_charge"]).dt.days.clip(0, 90)
df["duree_totale_j"] = (df["date_cloture"] - df["date_creation"]).dt.days.clip(0, 90)

print("Calcul des features agent...")
df["temps_passe_declare_min"] = pd.to_numeric(df["temps_passe_declare_min"], errors="coerce")
df["volume_dossiers"] = pd.to_numeric(df["volume_dossiers"], errors="coerce")

df["productivite_dossiers_par_heure"] = (
    df["volume_dossiers"] / (df["temps_passe_declare_min"] / 60)
).replace([np.inf, -np.inf], np.nan).clip(0, 25)

charge_jour = df.groupby(["matricule_agent", "date_creation"]).size().reset_index(name="charge_journaliere_nb_taches")
df = df.merge(charge_jour, on=["matricule_agent", "date_creation"], how="left")

moy_prod = df.groupby("matricule_agent")["productivite_dossiers_par_heure"].mean().reset_index(name="moy_productivite_agent")
df = df.merge(moy_prod, on="matricule_agent", how="left")

df["ecart_temps_declare_vs_reel"] = df["temps_passe_declare_min"] - df["duree_traitement_reelle_j"] * 480

df["date_entree"] = pd.to_datetime(df["date_entree"], errors="coerce")
df["anciennete_agent_mois"] = ((df["date_creation"] - df["date_entree"]).dt.days / 30.44).clip(0)

print("Calcul des features ETP...")
contrat_coeff = {
    "CDI": 1.0, "CDD": 0.85, "Interimaire": 0.75,
    "Prestataire": 0.80, "Alternant": 0.60, "Temps partiel": 0.50
}
df["coeff_temps_travail"] = df["type_contrat_agent"].map(contrat_coeff).fillna(0.85)
df["coeff_productivite_contrat"] = df["type_contrat_agent"].map(contrat_coeff).fillna(0.85)
df["contrat_actif_a_date"] = 1

absences["date_debut"] = pd.to_datetime(absences["date_debut"], errors="coerce")
absences["date_fin"] = pd.to_datetime(absences["date_fin"], errors="coerce")
absences_set = set()
for _, row in absences.iterrows():
    if pd.notna(row["date_debut"]) and pd.notna(row["date_fin"]):
        for d in pd.date_range(row["date_debut"], row["date_fin"]):
            absences_set.add((row["matricule"], d.date()))

df["est_absent"] = df.apply(
    lambda r: 1 if pd.notna(r["date_creation"]) and
    (r["matricule_agent"], r["date_creation"].date()) in absences_set else 0, axis=1
)

df["etp_agent_jour"] = df["coeff_temps_travail"] * df["contrat_actif_a_date"] * (1 - df["est_absent"])

etp_global = df.groupby("date_creation")["etp_agent_jour"].sum().reset_index(name="etp_total_disponible_jour")
df = df.merge(etp_global, on="date_creation", how="left")

etp_service = df.groupby(["date_creation", "service_agent"])["etp_agent_jour"].sum().reset_index(name="etp_disponible_service_jour")
df = df.merge(etp_service, on=["date_creation", "service_agent"], how="left")

print("Calcul des features metier...")
complexite_map = {"Simple": 1, "Moyen": 2, "Complexe": 3}
df["complexite_num"] = df["complexite"].map(complexite_map).fillna(2)

moy_temps_service = df.groupby("service_agent")["temps_passe_declare_min"].mean().reset_index(name="moy_temps_service")
df = df.merge(moy_temps_service, on="service_agent", how="left")
df["ratio_complexite"] = (df["temps_passe_declare_min"] / df["moy_temps_service"]).replace([np.inf, -np.inf], np.nan)

vol_global = df.groupby("date_creation")["volume_dossiers"].sum().reset_index(name="volume_total_entrant_jour")
df = df.merge(vol_global, on="date_creation", how="left")

vol_service = df.groupby(["date_creation", "service_agent"])["volume_dossiers"].sum().reset_index(name="volume_entrant_service_jour")
df = df.merge(vol_service, on=["date_creation", "service_agent"], how="left", suffixes=("", "_vs"))

df["charge_par_etp"] = (df["volume_total_entrant_jour"] / df["etp_total_disponible_jour"]).replace([np.inf, -np.inf], np.nan)
seuil_global = df["charge_par_etp"].quantile(0.75)
df["alerte_surcharge"] = (df["charge_par_etp"] > seuil_global).astype(int)

df["charge_par_etp_service"] = (df["volume_entrant_service_jour"] / df["etp_disponible_service_jour"]).replace([np.inf, -np.inf], np.nan)
seuil_service = df.groupby("service_agent")["charge_par_etp_service"].transform(lambda x: x.quantile(0.75))
df["alerte_surcharge_service"] = (df["charge_par_etp_service"] > seuil_service).astype(int)
df["seuil_alerte_service"] = seuil_service

print(f"  Features calculees : {df.shape[1]} colonnes")

# ------------------------------------------------------------------
# BLOC 3C : backtest agrege (split, verite terrain, prediction) --
# app/data/loader.py::charger_confusion_retard() fait un GROUP BY dessus,
# donc seules ces 3 colonnes comptent (pas besoin de l'identite des taches).
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("BLOC 3C : CLASSIFICATION DU RISQUE DE RETARD")
print("=" * 60)

model_path_3c = os.path.join("..", "models", "model_classification_retard.pkl")
if not os.path.exists(model_path_3c):
    model_path_3c = os.path.join("models", "model_classification_retard.pkl")

if os.path.exists(model_path_3c):
    model_dict_3c = joblib.load(model_path_3c)
    model_3c = model_dict_3c["model"]
    features_canonical_3c = model_dict_3c["features"]
    seuil = model_dict_3c.get("seuil_decision", 0.758)
    seuil_retard_j = model_dict_3c.get("seuil_retard_j", 3.0)
    date_fin_train = pd.Timestamp(model_dict_3c.get("date_fin_train"))
    date_fin_val = pd.Timestamp(model_dict_3c.get("date_fin_val"))
    print(f"  Modele charge : {model_path_3c} ({len(features_canonical_3c)} features, seuil={seuil:.3f})")
    print(f"  Coupures split : train<={date_fin_train.date()} val<={date_fin_val.date()}")

    df_enc = df.copy()
    for col, prefix in [("service_agent", "svc"), ("type_contrat_agent", "ctr"), ("type_processus", "proc")]:
        dummies = pd.get_dummies(df_enc[col], prefix=prefix).astype(int)
        df_enc = pd.concat([df_enc, dummies], axis=1)
    for feat in features_canonical_3c:
        if feat not in df_enc.columns:
            df_enc[feat] = 0

    # Verite terrain : seulement les taches closes (duree connue), meme
    # definition que modelisation_retard.py (P75 -> seuil_retard_j).
    df_enc["est_en_retard_reel"] = (df_enc["duree_traitement_reelle_j"] > seuil_retard_j).astype("Int64")

    df_3c = df_enc[df_enc["duree_traitement_reelle_j"].notna()].copy()
    df_3c = df_3c.dropna(subset=features_canonical_3c)

    proba = model_3c.predict_proba(df_3c[features_canonical_3c])[:, 1]
    df_3c["est_en_retard_predit"] = (proba >= seuil).astype(int)

    df_3c["split"] = np.select(
        [df_3c["date_creation"] <= date_fin_train, df_3c["date_creation"] <= date_fin_val],
        ["train", "validation"],
        default="test",
    )

    df_export = df_3c[["split", "est_en_retard_reel", "est_en_retard_predit"]].copy()
    df_export.to_sql("predictions_risque_retard", write_engine, if_exists="replace", index=False, chunksize=5000)
    print(f"  predictions_risque_retard : {len(df_export)} lignes ecrites (backtest complet, agregable)")
    print(df_export.groupby(["split", "est_en_retard_reel", "est_en_retard_predit"]).size())
else:
    print(f"  ATTENTION : modele non trouve ({model_path_3c})")

# ------------------------------------------------------------------
# BLOC 3D : detection d'anomalies -- TOUTES les taches scorees (pas
# seulement celles signalees), sinon charger_taux_anomalies_global()
# (AVG(est_anomalie) sur toute la table) donnerait 100%. Colonnes de base
# capitalisees pour matcher exactement le schema de features_detail que
# app/data/loader.py interroge (ID_Tache, Matricule_Agent, Service, ...).
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("BLOC 3D : DETECTION D'ANOMALIES")
print("=" * 60)

model_path_3d = os.path.join("..", "models", "model_isolation_forest.pkl")
if not os.path.exists(model_path_3d):
    model_path_3d = os.path.join("models", "model_isolation_forest.pkl")

if os.path.exists(model_path_3d):
    model_dict_3d = joblib.load(model_path_3d)
    model_3d = model_dict_3d["model"]
    scaler_3d = model_dict_3d.get("scaler")
    features_canonical_3d = model_dict_3d["features"]
    print(f"  Modele charge : {model_path_3d} ({len(features_canonical_3d)} features, scaler={'oui' if scaler_3d is not None else 'non'})")

    # Alias insensibles a la casse pour les features du modele (ex.
    # Temps_Passe_Declare_Min vs temps_passe_declare_min).
    lower_to_actual = {c.lower(): c for c in df.columns}
    for feat in features_canonical_3d:
        if feat not in df.columns and feat.lower() in lower_to_actual:
            df[feat] = df[lower_to_actual[feat.lower()]]

    features_ok_3d = [f for f in features_canonical_3d if f in df.columns]
    cols_identite = ["id_tache", "matricule_agent", "service_agent", "date_creation", "type_processus", "complexite"]
    df_3d = df[cols_identite + features_ok_3d].dropna(subset=features_ok_3d).copy()

    X_3d = df_3d[features_ok_3d]
    if scaler_3d is not None:
        X_3d = scaler_3d.transform(X_3d)

    scores = model_3d.score_samples(X_3d)
    predictions = model_3d.predict(X_3d)
    df_3d["score_anomalie"] = scores
    df_3d["est_anomalie"] = (predictions == -1).astype(int)

    # Renommage vers le schema attendu par loader.py (memes noms/casse que
    # features_detail : colonnes de base capitalisees, features en minuscules).
    df_export_3d = df_3d.rename(columns={
        "id_tache": "ID_Tache", "matricule_agent": "Matricule_Agent",
        "service_agent": "Service", "date_creation": "Date_Creation",
        "type_processus": "Type_Processus", "complexite": "Complexite",
        "temps_passe_declare_min": "Temps_Passe_Declare_Min",
        "volume_dossiers": "Volume_Dossiers",
    })

    # Colonnes utiles uniquement au detail des 200 pires anomalies (page
    # Derive) : creuses hors des lignes est_anomalie=1 (~2,7% des lignes),
    # la requete d'agregation (Matricule_Agent/Service/score/flag) n'a besoin
    # que des lignes -- pas de ces colonnes de detail -- pour tout le monde.
    colonnes_detail_seules = ["ID_Tache", "Date_Creation", "Type_Processus", "Complexite",
                               "Temps_Passe_Declare_Min", "Volume_Dossiers", "productivite_dossiers_par_heure"]
    colonnes_detail_presentes = [c for c in colonnes_detail_seules if c in df_export_3d.columns]
    df_export_3d.loc[df_export_3d["est_anomalie"] == 0, colonnes_detail_presentes] = np.nan

    df_export_3d.to_sql("anomalies_detectees", write_engine, if_exists="replace", index=False, chunksize=5000)
    print(f"  anomalies_detectees : {len(df_export_3d)} lignes ecrites (toutes les taches scorees)")
    print(f"  Taux d'anomalies : {df_export_3d['est_anomalie'].mean()*100:.1f}%")
else:
    print(f"  ATTENTION : modele non trouve ({model_path_3d})")

# ------------------------------------------------------------------
# features_detail (reduite) : seules les colonnes utilisees par la page
# Derive (agregation par agent dans loader.py::charger_donnees) -- pas les
# 57 colonnes de la vraie features_detail, qui ne tient pas dans le quota
# Neon gratuit.
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("FEATURES_DETAIL (version reduite pour la page Derive)")
print("=" * 60)

# Union des colonnes reellement lues dans app/data/loader.py : df_feat
# (charger_donnees), charger_historique_agent, charger_taches_en_cours +
# FEATURES_NUM_RETARD. Capitalisation des colonnes de base identique a la
# vraie features_detail (feature_engineering.py) ; features calculees en
# minuscules, inchangees.
identite_renamed = {
    "id_tache": "ID_Tache", "matricule_agent": "Matricule_Agent",
    "service": "Service", "service_agent": "Service_Agent",
    "type_contrat": "Type_Contrat", "type_contrat_agent": "Type_Contrat_Agent",
    "type_processus": "Type_Processus", "date_creation": "Date_Creation",
    "complexite": "Complexite", "statut": "Statut",
}
colonnes_calculees = [
    "productivite_dossiers_par_heure", "duree_traitement_reelle_j",
    "delai_prise_en_charge_j", "anciennete_agent_mois",
    "charge_par_etp_service", "est_absent", "complexite_num",
    "coeff_productivite_contrat", "charge_journaliere_nb_taches",
    "etp_agent_jour", "etp_disponible_service_jour", "jour_semaine",
    "mois", "trimestre", "semaine_du_mois", "est_fin_de_mois",
    "est_fin_trimestre", "est_lundi", "est_vendredi", "alerte_surcharge_service",
]
colonnes_derive = list(identite_renamed.keys()) + colonnes_calculees
df_derive = df[colonnes_derive].dropna(subset=["productivite_dossiers_par_heure"]).rename(columns=identite_renamed)

# Colonnes utiles UNIQUEMENT a charger_taches_en_cours() (page Surcharge,
# scoring des taches "En cours" recentes) : les rendre creuses (NaN hors de
# ce sous-ensemble) plutot que remplies sur les ~895k lignes -- Postgres
# stocke un NULL en ~1 bit, contre la valeur reelle sinon. Le reste
# (Matricule_Agent, Service_Agent, Type_Contrat_Agent, Date_Creation,
# Complexite, Statut, productivite/duree/delai/anciennete/charge/absence)
# reste dense : necessaire sur toutes les lignes pour df_feat et
# charger_historique_agent.
colonnes_taches_en_cours_seules = [
    "ID_Tache", "Service", "Type_Contrat", "Type_Processus",
    "complexite_num", "coeff_productivite_contrat", "charge_journaliere_nb_taches",
    "etp_agent_jour", "etp_disponible_service_jour", "jour_semaine", "mois",
    "trimestre", "semaine_du_mois", "est_fin_de_mois", "est_fin_trimestre",
    "est_lundi", "est_vendredi", "alerte_surcharge_service",
]
date_min_en_cours = df_derive["Date_Creation"].max() - pd.Timedelta(days=FENETRE_TACHES_EN_COURS_JOURS)
masque_en_cours = (df_derive["Statut"] == "En cours") & (df_derive["Date_Creation"] >= date_min_en_cours)
df_derive.loc[~masque_en_cours, colonnes_taches_en_cours_seules] = np.nan
print(f"  Colonnes 'taches en cours' rendues creuses hors des {masque_en_cours.sum()} taches concernees")

df_derive.to_sql("features_detail", write_engine, if_exists="replace", index=False, chunksize=5000)
print(f"  features_detail (reduite) : {len(df_derive)} lignes, {df_derive.shape[1]} colonnes ecrites")

print("\n" + "=" * 60)
print("Blocs 3C, 3D et features_detail (reduite) termines avec succes.")
print("=" * 60)
