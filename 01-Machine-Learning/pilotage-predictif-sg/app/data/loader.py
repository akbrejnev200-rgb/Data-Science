"""
loader.py

Chargement unique (au demarrage du serveur) des donnees et modeles pour
l'application Dash. La quasi-totalite des predictions/scores affiches par
l'app sont deja pre-calcules dans les tables predictions_*/anomalies_detectees
par la Phase 3 du pipeline PostgreSQL -- les modeles .pkl ne sont charges
que pour exposer leurs metadonnees (MAE, accuracy, seuils) a la page
Fiabilite.

Quatre exceptions :
- charger_historique_agent() est appelee a la demande (callback de
  selection d'agent, page Derive), bornee a un seul agent.
- predire_charge_globale() appelle reellement m_charge.predict() pour le
  prochain jour ouvre (page Surcharge) : necessaire car aucune table de
  prevision de charge par ETP n'existe (Bloc 3B n'a produit qu'un backtest
  historique).
- scorer_risque_retard() appelle m_retard.predict_proba() sur les taches
  actuellement "En cours" (page Surcharge) : necessaire car
  predictions_risque_retard n'est qu'un backtest sur taches deja closes,
  jamais sur des taches encore ouvertes.
- charger_incidents()/inserer_incident()/supprimer_incident() (page
  Incidents & Annonces) lisent/ecrivent incidents_manager en direct a
  chaque saisie du manager -- seule partie de l'app qui ecrit dans
  PostgreSQL plutot que de se contenter d'y lire des donnees deja calculees.
"""

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "database"))
from db_connection import get_engine, MODELS_DIR

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

NOM_JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
NOM_MOIS = ["", "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
            "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]

_engine = get_engine()

# TODAY = derniere date reelle disponible dans les donnees, PAS la date du
# jour calendaire : celle-ci ne recoupe ni l'historique ni les previsions
# (ecart de plusieurs semaines entre la fin des donnees reelles et le debut
# des previsions J+1 a J+30), et laisserait toutes les cartes "aujourd'hui"
# vides si on utilisait datetime.date.today().
TODAY = pd.read_sql("SELECT MAX(date) AS d FROM features_journalier", _engine)["d"].iloc[0].date()


def charger_donnees():
    """Un seul appel au demarrage : retourne (dj, djs, df_feat, df_anomalies, df_previsions)."""
    dj = pd.read_sql("SELECT * FROM features_journalier ORDER BY date", _engine)

    djs = pd.read_sql('SELECT * FROM features_journalier_service ORDER BY date, "Service"', _engine)

    df_feat = pd.read_sql("""
        SELECT "Matricule_Agent", "Service_Agent", "Type_Contrat_Agent",
               COUNT(*)                                     AS nb_taches,
               AVG(productivite_dossiers_par_heure)          AS productivite_moy,
               AVG(duree_traitement_reelle_j)                 AS duree_traitement_moy_j,
               AVG(delai_prise_en_charge_j)                    AS delai_prise_en_charge_moy_j,
               MAX(anciennete_agent_mois)                      AS anciennete_mois,
               AVG(charge_par_etp_service)                     AS charge_par_etp_moy,
               SUM(est_absent)                                 AS jours_absence
        FROM features_detail
        GROUP BY "Matricule_Agent", "Service_Agent", "Type_Contrat_Agent"
    """, _engine)

    df_anomalies = {
        "par_agent": pd.read_sql("""
            SELECT "Matricule_Agent", "Service",
                   COUNT(*)            AS nb_taches_evaluees,
                   SUM(est_anomalie)   AS nb_anomalies,
                   AVG(score_anomalie) AS score_moyen,
                   MIN(score_anomalie) AS score_min
            FROM anomalies_detectees
            GROUP BY "Matricule_Agent", "Service"
        """, _engine),
        "top_anomalies": pd.read_sql("""
            SELECT "ID_Tache", "Matricule_Agent", "Service", "Date_Creation", "Type_Processus",
                   "Complexite", "Temps_Passe_Declare_Min", "Volume_Dossiers",
                   productivite_dossiers_par_heure, score_anomalie
            FROM anomalies_detectees
            WHERE est_anomalie = 1
            ORDER BY score_anomalie ASC
            LIMIT 200
        """, _engine),
    }

    df_previsions = pd.read_sql("SELECT * FROM predictions_volumes_j30 ORDER BY date", _engine)

    return dj, djs, df_feat, df_anomalies, df_previsions


def charger_modeles():
    """Charge les 4 .pkl. Ne sert qu'a exposer leurs metadonnees (MAE, accuracy,
    seuils) a la page Fiabilite -- aucun .predict() n'est jamais appele."""
    m_volumes = joblib.load(MODELS_DIR / "model_volumes.pkl")
    m_charge = joblib.load(MODELS_DIR / "model_charge_etp.pkl")
    m_retard = joblib.load(MODELS_DIR / "model_classification_retard.pkl")
    m_anomaly = joblib.load(MODELS_DIR / "model_isolation_forest.pkl")
    return m_volumes, m_charge, m_retard, m_anomaly


def charger_taux_anomalies_global():
    """Taux d'anomalies sur l'ensemble des taches evaluees -- calcule depuis
    anomalies_detectees (donnee de scoring, rafraichie par
    predict_anomalies.py), pas depuis le .pkl (metadonnee de fit, retiree
    lors de la separation fit/predict)."""
    return pd.read_sql("SELECT AVG(est_anomalie) AS taux FROM anomalies_detectees", _engine)["taux"].iloc[0]


def charger_predictions_charge():
    """Backtest train/validation/test du modele de charge par ETP (Bloc 3B)."""
    return pd.read_sql(
        "SELECT date, split, charge_reelle, charge_predite, modele FROM predictions_charge_etp ORDER BY date",
        _engine,
    )


def charger_confusion_retard():
    """Matrice de confusion agregee, ventilee par split (train/validation/
    test -- colonne desormais presente dans predictions_risque_retard,
    generee par predict_retard.py), sans jamais charger les 717k lignes
    brutes. Filtrable cote app, meme pattern que charger_predictions_charge()."""
    return pd.read_sql("""
        SELECT split, est_en_retard_predit, est_en_retard_reel, COUNT(*) AS n
        FROM predictions_risque_retard
        GROUP BY split, est_en_retard_predit, est_en_retard_reel
    """, _engine)


# ----------------------------------------------------------------------
# Incidents & Annonces (page /incidents) -- seule ecriture PostgreSQL de
# l'app. incidents_manager alimente aussi, en lecture, les regresseurs
# exogenes nb_etp_impactes_jour / volume_annonce_jour des Blocs 3A/3B (cf.
# modelisation_common.calculer_features_incidents, cote pipeline).
# ----------------------------------------------------------------------

def charger_incidents(limite=30):
    """Les `limite` derniers evenements saisis (incidents ETP + annonces),
    du plus recent au plus ancien -- alimente le tableau d'historique en
    bas de la page /incidents."""
    return pd.read_sql(
        "SELECT * FROM incidents_manager ORDER BY date_saisie DESC LIMIT %(limite)s",
        _engine, params={"limite": limite}, parse_dates=["date_evenement", "date_saisie"],
    )


def inserer_incident(date_evenement, service, type_evenement, motif=None, nb_etp_impactes=None,
                      duree_jours=1, volume_supplementaire_estime=None, commentaire=None):
    """Ecrit un evenement dans incidents_manager. duree_jours vaut la duree
    estimee saisie directement (incident_etp) ou (date_fin - date_debut + 1)
    deja calculee par l'appelant (annonce) -- la table ne stocke qu'une
    date de debut (date_evenement) et une duree, jamais de date de fin."""
    with _engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO incidents_manager
                (date_evenement, service, type_evenement, motif, nb_etp_impactes,
                 duree_jours, volume_supplementaire_estime, commentaire)
            VALUES
                (:date_evenement, :service, :type_evenement, :motif, :nb_etp_impactes,
                 :duree_jours, :volume_supplementaire_estime, :commentaire)
        """), {
            "date_evenement": date_evenement, "service": service, "type_evenement": type_evenement,
            "motif": motif, "nb_etp_impactes": nb_etp_impactes, "duree_jours": duree_jours,
            "volume_supplementaire_estime": volume_supplementaire_estime, "commentaire": commentaire,
        })


def supprimer_incident(id_incident):
    with _engine.begin() as conn:
        conn.execute(text("DELETE FROM incidents_manager WHERE id = :id"), {"id": id_incident})


def charger_historique_agent(matricule):
    """Seule requete SQL executee apres le demarrage du serveur, a la demande
    (callback de selection d'agent, page Derive) -- bornee a un seul agent
    plutot que de charger les 939k lignes de features_detail en memoire."""
    return pd.read_sql("""
        SELECT "Date_Creation", productivite_dossiers_par_heure, duree_traitement_reelle_j,
               "Complexite", "Statut"
        FROM features_detail
        WHERE "Matricule_Agent" = %(matricule)s
        ORDER BY "Date_Creation"
    """, _engine, params={"matricule": matricule})


def etendre_djs_avec_previsions(djs, df_previsions):
    """Ne fusionne JAMAIS la prevision globale dans les lignes par service :
    aucune prevision par service n'existe (le Bloc 3A ne prevoit que le volume
    total de la banque). djs reste structurellement inchange ; c'est la page
    Surcharge qui affiche la prevision globale comme une serie separee et
    clairement etiquetee, visible uniquement quand 'Tous services' est
    selectionne dans le filtre."""
    djs = djs.copy()
    djs["prevision_disponible"] = False
    return djs


# ----------------------------------------------------------------------
# ETP projete (prochain jour ouvre) -- reprend la logique du Bloc C de
# feature_engineering.py (contrat actif + absence reelle), generalisee a
# une date cible quelconque, avec un taux d'absence de repli par type de
# contrat quand aucune donnee de conge ne couvre cette date (cf. echange
# avec l'utilisateur : ne jamais supposer 100% de presence par defaut).
# ----------------------------------------------------------------------

def convertir_temps_travail(val):
    """Identique a la fonction du meme nom dans feature_engineering.py."""
    if pd.isna(val):
        return 1.0
    val = str(val).strip().replace("%", "")
    try:
        v = float(val)
        return v / 100 if v > 1 else v
    except ValueError:
        if "plein" in val.lower():
            return 1.0
        if "partiel" in val.lower():
            return 0.8
        return 1.0


MAPPING_CONTRAT = {
    "CDI": "CDI", "cdi": "CDI",
    "CDD": "CDD", "cdd": "CDD",
    "Intérimaire": "Intérimaire", "Interimaire": "Intérimaire", "INTERIMAIRE": "Intérimaire",
    "Alternant": "Alternant", "alternant": "Alternant",
    "Prestataire": "Prestataire", "PRESTATAIRE": "Prestataire",
    "Temps partiel": "Temps partiel",
}


def charger_agents_absences():
    """Roster agents (contrats, temps de travail) et conges/absences reels
    -- tables cleanes, petites (~100 et ~6000 lignes), chargees une fois
    au demarrage pour permettre le calcul d'un ETP futur reel plutot
    qu'estime, tant que la couverture des conges le permet.

    Normalise ici le type_contrat de absences_clean (CDI/cdi, INTERIMAIRE/
    Interimaire/Intérimaire...) : data_cleaning.py ne l'appliquait que sur
    taches_clean et agents_clean, pas sur absences_clean -- fige tant que
    le pipeline n'a pas ete rejoue avec le correctif."""
    df_agents = pd.read_sql("SELECT * FROM agents_clean", _engine)
    df_absences = pd.read_sql(
        "SELECT matricule, type_contrat, service, type_absence, date_debut, date_fin, duree_jours "
        "FROM absences_clean",
        _engine, parse_dates=["date_debut", "date_fin"],
    )
    df_agents["date_entree"] = pd.to_datetime(df_agents["date_entree"], errors="coerce")
    df_agents["date_fin_contrat"] = pd.to_datetime(df_agents["date_fin_contrat"], errors="coerce")
    df_absences["type_contrat"] = (
        df_absences["type_contrat"].str.strip().map(MAPPING_CONTRAT).fillna("Non renseigné")
    )
    return df_agents, df_absences


def prochain_jour_ouvre(date_ref):
    """Lundi a vendredi uniquement (aucun jour ferie modelise, coherent
    avec le calendrier deja utilise pour les previsions de volume)."""
    d = pd.Timestamp(date_ref) + timedelta(days=1)
    while d.dayofweek >= 5:
        d += timedelta(days=1)
    return d


PROCHAIN_JOUR_OUVRE = prochain_jour_ouvre(TODAY)


def prochains_jours_ouvres(date_ref, n=7):
    jours = []
    d = pd.Timestamp(date_ref)
    for _ in range(n):
        d = prochain_jour_ouvre(d)
        jours.append(d)
    return jours


PROCHAINS_JOURS_OUVRES = prochains_jours_ouvres(TODAY, 30)


def calculer_taux_absence_repli(df_agents, df_absences):
    """Taux d'absence moyen par type de contrat (jours d'absence / jours de
    contrat actif, sur tout l'historique) -- sert de repli quand aucun conge
    n'est enregistre pour la date cible, au lieu de supposer 100% de
    presence par defaut. Ne capture donc que l'absenteisme moyen (conges,
    maladie, formation, semaines ecole des alternants...), pas un jour
    precis."""
    agents = df_agents.dropna(subset=["date_entree"]).copy()
    borne_fin = pd.Timestamp(TODAY)
    date_fin_effective = agents["date_fin_contrat"].fillna(borne_fin).clip(upper=borne_fin)
    jours_actifs = (date_fin_effective - agents["date_entree"]).dt.days.clip(lower=0) + 1
    jours_actifs_par_contrat = jours_actifs.groupby(agents["type_contrat"]).sum()

    jours_absence_par_contrat = df_absences.groupby("type_contrat")["duree_jours"].sum()

    taux = (jours_absence_par_contrat / jours_actifs_par_contrat).fillna(0.1).clip(0, 0.9)
    return taux.to_dict()


def etp_disponible_projete(date_cible, df_agents, df_absences, taux_repli):
    """ETP disponible (global + par service) pour une date cible. Utilise
    les conges reellement enregistres si la date est couverte par
    absences_clean ; sinon applique le taux d'absence de repli par type de
    contrat. Retourne (etp_global, etp_par_service: Series, couverture_reelle: bool)."""
    date_cible = pd.Timestamp(date_cible)
    couverture_reelle = date_cible <= df_absences["date_fin"].max()

    agents = df_agents.dropna(subset=["date_entree"]).copy()
    agents = agents[
        (agents["date_entree"] <= date_cible) &
        (agents["date_fin_contrat"].isna() | (agents["date_fin_contrat"] >= date_cible))
    ].copy()
    agents["coeff_temps_travail"] = agents["temps_travail"].apply(convertir_temps_travail)

    if couverture_reelle:
        matricules_absents = df_absences.loc[
            (df_absences["date_debut"] <= date_cible) & (df_absences["date_fin"] >= date_cible),
            "matricule",
        ].unique()
        agents["taux_presence"] = (~agents["matricule"].isin(matricules_absents)).astype(float)
    else:
        agents["taux_presence"] = 1 - agents["type_contrat"].map(taux_repli).fillna(0.1)

    agents["etp_agent_jour"] = agents["coeff_temps_travail"] * agents["taux_presence"]

    etp_global = agents["etp_agent_jour"].sum()
    etp_par_service = agents.dropna(subset=["service"]).groupby("service")["etp_agent_jour"].sum()
    return etp_global, etp_par_service, couverture_reelle


FEATURES_CHARGE = [
    "volume_lag_1j", "volume_lag_7j", "volume_lag_14j", "volume_lag_30j",
    "etp_lag_1j", "etp_lag_7j", "etp_lag_14j", "etp_lag_30j",
    "volume_moy_7j", "charge_moy_7j",
    "jour_semaine", "mois", "trimestre", "semaine_du_mois",
    "est_fin_de_mois", "est_fin_trimestre", "est_lundi", "est_vendredi",
    "etp_disponible", "taux_complexes",
    "jour_semaine_sin", "jour_semaine_cos", "mois_sin", "mois_cos",
    "nb_etp_impactes_jour", "volume_annonce_jour",
]
# Colonnes disponibles au maximum -- ce que predire_charge_globale() sait
# calculer. Le modele reellement charge (m_charge["features"], fige au
# moment de son entrainement par modelisation_charge_etp.py) peut en
# attendre un sous-ensemble plus ancien (ex. avant l'ajout de
# nb_etp_impactes_jour/volume_annonce_jour) : on filtre toujours sur
# m_charge["features"] plutot que sur cette liste, pour rester compatible
# avec un .pkl pas encore reentraine.


def _features_incidents_jour(date_cible):
    """Equivalent, borne a une seule date, de
    modelisation_common.calculer_features_incidents() (cote pipeline) :
    somme des ETP impactes / du volume annonce des evenements
    incidents_manager actifs a `date_cible`. Duplique plutot qu'importe
    depuis notebooks_avec_ecriture_dans_database/ -- app/ et le pipeline
    restent deux mondes independants (meme principe que
    convertir_temps_travail(), duplique depuis feature_engineering.py)."""
    date_cible = pd.Timestamp(date_cible)
    df_inc = pd.read_sql("SELECT * FROM incidents_manager", _engine, parse_dates=["date_evenement"])
    if not len(df_inc):
        return 0.0, 0.0
    duree = pd.to_numeric(df_inc["duree_jours"], errors="coerce").fillna(1).clip(lower=1)
    date_fin_effective = df_inc["date_evenement"] + pd.to_timedelta(duree - 1, unit="D")
    actif = (df_inc["date_evenement"] <= date_cible) & (date_fin_effective >= date_cible)
    nb_etp = df_inc.loc[actif & (df_inc["type_evenement"] == "incident_etp"), "nb_etp_impactes"].sum()
    volume = df_inc.loc[actif & (df_inc["type_evenement"] == "annonce"), "volume_supplementaire_estime"].sum()
    return float(nb_etp), float(volume)


def predire_charge_globale(m_charge, dj, date_cible, etp_global_prevu):
    """Seule inference ML en direct de l'app : appelle m_charge.predict()
    pour le prochain jour ouvre. Les lags (volume/ETP a J-1/7/14/30) sont
    lus par position dans l'historique trie (memes semantiques que le
    .shift() utilise a l'entrainement, cf. feature_engineering.py Bloc E) --
    valide uniquement pour predire le jour immediatement apres le dernier
    jour reel, pas un horizon plus lointain."""
    date_cible = pd.Timestamp(date_cible)
    dj_sorted = dj.sort_values("date").reset_index(drop=True)

    def lag(col, n):
        return dj_sorted[col].iloc[-n] if len(dj_sorted) >= n else np.nan

    jour_semaine = date_cible.dayofweek
    mois = date_cible.month
    nb_etp_impactes_jour, volume_annonce_jour = _features_incidents_jour(date_cible)
    ligne = pd.DataFrame([{
        "volume_lag_1j": lag("volume_entrant_jour", 1),
        "volume_lag_7j": lag("volume_entrant_jour", 7),
        "volume_lag_14j": lag("volume_entrant_jour", 14),
        "volume_lag_30j": lag("volume_entrant_jour", 30),
        "etp_lag_1j": lag("etp_disponible", 1),
        "etp_lag_7j": lag("etp_disponible", 7),
        "etp_lag_14j": lag("etp_disponible", 14),
        "etp_lag_30j": lag("etp_disponible", 30),
        "volume_moy_7j": dj_sorted["volume_entrant_jour"].tail(7).mean(),
        "charge_moy_7j": dj_sorted["charge_par_etp"].tail(7).mean(),
        "jour_semaine": jour_semaine,
        "mois": mois,
        "trimestre": date_cible.quarter,
        "semaine_du_mois": ((date_cible.day - 1) // 7) + 1,
        "est_fin_de_mois": int(date_cible.day >= 25),
        "est_fin_trimestre": int(mois in [3, 6, 9, 12]),
        "est_lundi": int(jour_semaine == 0),
        "est_vendredi": int(jour_semaine == 4),
        "etp_disponible": etp_global_prevu,
        "taux_complexes": dj_sorted["taux_complexes"].tail(30).mean(),
        "jour_semaine_sin": np.sin(2 * np.pi * jour_semaine / 7),
        "jour_semaine_cos": np.cos(2 * np.pi * jour_semaine / 7),
        "mois_sin": np.sin(2 * np.pi * mois / 12),
        "mois_cos": np.cos(2 * np.pi * mois / 12),
        "nb_etp_impactes_jour": nb_etp_impactes_jour,
        "volume_annonce_jour": volume_annonce_jour,
    }])[m_charge["features"]]

    if m_charge["scaler"] is not None:
        X = m_charge["scaler"].transform(ligne)
    else:
        X = ligne
    return float(m_charge["model"].predict(X)[0])


def construire_serie_prevision(dj, df_previsions, df_agents, df_absences, taux_repli, m_charge, jours_cibles):
    """Prevision multi-jours (J+1 a J+7) : chaque jour est reprevu en
    etendant la serie de lags avec le jour precedent (reel s'il est deja
    connu, sinon la valeur qu'on vient de prevoir juste avant) -- prevision
    recursive, comme le ferait un pipeline qui tourne chaque jour. L'ETP
    reste calcule independamment pour chaque jour cible (contrats + conges
    reels tant que couverts, sinon taux de repli). Retourne une liste de
    dicts (un par horizon), dans le meme ordre que jours_cibles.

    En plus de la prevision "modele" (charge_prevue, volume_prevu), calcule
    un ajustement affichable tout de suite, sans reentrainement : si un
    incident ETP ou une annonce (incidents_manager) est actif ce jour-la,
    charge_prevue_ajustee reappelle le meme modele avec un ETP disponible
    reduit du nombre d'ETP signales, et volume_prevu_ajuste ajoute le
    volume annonce au volume prevu -- un simple recalcul transparent
    (jamais un nouvel entrainement), sur le meme principe que les outils de
    gestion de charge qui distinguent toujours "prevision modele" et
    "prevision ajustee des evenements connus". La chaine recursive de lags
    (`serie`) continue de s'appuyer sur les valeurs modele non ajustees,
    pour ne jamais laisser un ajustement manuel se propager en cascade
    dans les jours suivants."""
    serie = dj[["date", "volume_entrant_jour", "etp_disponible", "charge_par_etp", "taux_complexes"]].copy()
    serie = serie.sort_values("date").reset_index(drop=True)

    resultats = []
    for h, date_cible in enumerate(jours_cibles, start=1):
        ligne_prevue = df_previsions[df_previsions["horizon_j"] == h]
        volume_prevu = ligne_prevue["volume_prevu"].iloc[0] if len(ligne_prevue) else None
        volume_min = ligne_prevue["volume_prevu_min"].iloc[0] if len(ligne_prevue) else None
        volume_max = ligne_prevue["volume_prevu_max"].iloc[0] if len(ligne_prevue) else None

        etp_global, etp_service, couverture_reelle = etp_disponible_projete(
            date_cible, df_agents, df_absences, taux_repli,
        )
        charge_prevue = predire_charge_globale(m_charge, serie, date_cible, etp_global) if etp_global else None

        nb_etp_impactes_jour, volume_annonce_jour = _features_incidents_jour(date_cible)

        volume_prevu_ajuste = None if volume_prevu is None else volume_prevu + volume_annonce_jour
        if nb_etp_impactes_jour and etp_global:
            etp_ajuste = max(etp_global - nb_etp_impactes_jour, 0.1)
            charge_prevue_ajustee = predire_charge_globale(m_charge, serie, date_cible, etp_ajuste)
        else:
            charge_prevue_ajustee = charge_prevue

        resultats.append({
            "horizon_j": h, "date": date_cible,
            "volume_prevu": volume_prevu, "volume_prevu_min": volume_min, "volume_prevu_max": volume_max,
            "etp_global": etp_global, "etp_service": etp_service, "couverture_reelle": couverture_reelle,
            "charge_prevue": charge_prevue,
            "nb_etp_impactes_jour": nb_etp_impactes_jour, "volume_annonce_jour": volume_annonce_jour,
            "volume_prevu_ajuste": volume_prevu_ajuste, "charge_prevue_ajustee": charge_prevue_ajustee,
        })

        serie = pd.concat([serie, pd.DataFrame([{
            "date": date_cible, "volume_entrant_jour": volume_prevu,
            "etp_disponible": etp_global, "charge_par_etp": charge_prevue,
            "taux_complexes": serie["taux_complexes"].tail(30).mean(),
        }])], ignore_index=True)

    return resultats


# ----------------------------------------------------------------------
# Risque de retard -- scoring en direct des taches "En cours" (page
# Surcharge). Reprend exactement les features et l'encodage one-hot du
# Bloc 3C (modelisation_retard.py) pour rester coherent avec le modele
# deja entraine.
# ----------------------------------------------------------------------

FEATURES_NUM_RETARD = [
    "complexite_num", "coeff_productivite_contrat",
    "charge_journaliere_nb_taches", "etp_agent_jour",
    "etp_disponible_service_jour", "charge_par_etp_service",
    "anciennete_agent_mois", "delai_prise_en_charge_j",
    "jour_semaine", "mois", "trimestre", "semaine_du_mois",
    "est_fin_de_mois", "est_fin_trimestre", "est_lundi", "est_vendredi",
    "est_absent", "alerte_surcharge_service",
]


FENETRE_TACHES_EN_COURS_JOURS = 30  # au-dela, "En cours" reflete des taches figees
# (artefact du jeu de donnees synthetique -- age median ~394 jours sur
# l'ensemble des taches "En cours"), pas un vrai retard actionnable.


def charger_taches_en_cours():
    """Taches actuellement ouvertes (Statut = 'En cours') et creees dans
    les FENETRE_TACHES_EN_COURS_JOURS derniers jours, avec les features
    necessaires au scoring de risque de retard. Restreint aux taches
    recentes : la table contient aussi des taches "En cours" vieilles de
    plusieurs centaines de jours (artefact du jeu de donnees synthetique),
    qui rendraient la liste de triage inexploitable."""
    colonnes_num = ", ".join(FEATURES_NUM_RETARD)
    date_min = pd.Timestamp(TODAY) - timedelta(days=FENETRE_TACHES_EN_COURS_JOURS)
    return pd.read_sql(f"""
        SELECT "ID_Tache", "Matricule_Agent", "Service", "Type_Contrat", "Type_Processus",
               "Date_Creation", {colonnes_num}
        FROM features_detail
        WHERE "Statut" = 'En cours' AND "Date_Creation" >= %(date_min)s
    """, _engine, params={"date_min": date_min}, parse_dates=["Date_Creation"])


def scorer_risque_retard(m_retard, df_taches):
    """Seule inference ML en direct pour ce modele : predict_proba() sur
    les taches ouvertes, avec le meme encodage one-hot Service/Type_Contrat/
    Type_Processus qu'a l'entrainement (colonnes manquantes = 0, alignees
    sur m_retard['features']). Les taches sans delai_prise_en_charge_j
    (pas encore prises en charge par un agent) sont exclues -- une feature
    du modele leur manque, on ne peut pas encore les evaluer."""
    df = df_taches.dropna(subset=FEATURES_NUM_RETARD).copy()
    if not len(df):
        return df.assign(score_risque_retard=pd.Series(dtype=float), est_a_risque=pd.Series(dtype=int))

    df_encode = pd.get_dummies(df, columns=["Service", "Type_Contrat", "Type_Processus"],
                                prefix=["svc", "ctr", "proc"])
    X = df_encode.reindex(columns=m_retard["features"], fill_value=0)

    proba = m_retard["model"].predict_proba(X)[:, 1]
    df["score_risque_retard"] = proba
    df["est_a_risque"] = (proba >= m_retard["seuil_decision"]).astype(int)
    return df
