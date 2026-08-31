"""
modelisation_common.py

Utilitaire pur partage par les scripts de la Phase 3 (3A/3B/3C/3D) : split
temporel 70/15/15 et calcul des regresseurs exogenes issus des incidents/
annonces saisis manuellement (page /incidents de l'app). Ne contient aucune
donnee ni etat propre : chaque script de modelisation reste independant et
n'echange aucune variable en memoire avec les autres (meme principe que
db_connection.get_engine()) -- ajouter_features_incidents() recoit son
propre `engine` en parametre plutot que d'en creer un. Les fonctions de
calcul de metriques restent locales a chaque script, leur definition
differant d'un bloc a l'autre (regression vs classification).
"""

import pandas as pd


def split_temporel(df, frac_train=0.70, frac_val=0.15):
    """Split chronologique 70/15/15 (jamais de split aleatoire sur une serie temporelle)."""
    n = len(df)
    n_train = int(n * frac_train)
    n_val = int(n * frac_val)
    return df.iloc[:n_train].copy(), df.iloc[n_train:n_train + n_val].copy(), df.iloc[n_train + n_val:].copy()


def calculer_features_incidents(dates, engine):
    """Pour chaque date de `dates` (iterable de dates/Timestamps), calcule
    nb_etp_impactes_jour (somme des ETP impactes par les incidents_etp actifs
    ce jour) et volume_annonce_jour (somme du volume supplementaire des
    annonces actives ce jour), a partir de incidents_manager. Un evenement
    est actif sur [date_evenement, date_evenement + duree_jours - 1].

    Retourne un DataFrame avec une colonne "date" (une ligne par valeur
    unique de `dates`) et les deux colonnes calculees, prete a etre
    fusionnee via merge(..., on="date"). Si incidents_manager est encore
    vide (table jamais alimentee) ou inexistante, retourne des zeros plutot
    que d'echouer -- les deux regresseurs restent valides des le premier
    jour, avant toute saisie manager."""
    dates_uniques = pd.to_datetime(pd.Series(dates).unique())
    out = pd.DataFrame({
        "date": dates_uniques,
        "nb_etp_impactes_jour": 0.0,
        "volume_annonce_jour": 0.0,
    })

    try:
        df_inc = pd.read_sql("SELECT * FROM incidents_manager", engine, parse_dates=["date_evenement"])
    except Exception:
        return out
    if not len(df_inc):
        return out

    df_inc["duree_jours"] = pd.to_numeric(df_inc["duree_jours"], errors="coerce").fillna(1).clip(lower=1)
    df_inc["date_fin_effective"] = df_inc["date_evenement"] + pd.to_timedelta(df_inc["duree_jours"] - 1, unit="D")

    for _, inc in df_inc.iterrows():
        actif = (out["date"] >= inc["date_evenement"]) & (out["date"] <= inc["date_fin_effective"])
        if not actif.any():
            continue
        if inc["type_evenement"] == "incident_etp":
            out.loc[actif, "nb_etp_impactes_jour"] += inc["nb_etp_impactes"] or 0
        elif inc["type_evenement"] == "annonce":
            out.loc[actif, "volume_annonce_jour"] += inc["volume_supplementaire_estime"] or 0

    return out
