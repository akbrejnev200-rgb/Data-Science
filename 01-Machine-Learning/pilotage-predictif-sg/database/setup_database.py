"""
setup_database.py

Cree (si necessaire) les tables STAGING (bronze) et CLEAN (silver).
Les tables FEATURES (gold) ne sont pas creees ici : elles sont creees
automatiquement par pandas.to_sql(if_exists="replace") dans
feature_engineering.py, car leur schema (colonnes calculees, nombreuses)
est directement derive du code plutot que fige a l'avance.

Ce script est idempotent : il peut etre relance plusieurs fois sans
creer de doublons ni provoquer d'erreur, grace a IF NOT EXISTS.

Usage :
    python setup_database.py
"""

from sqlalchemy import text
from db_connection import get_engine

engine = get_engine()

CREATE_TABLES_SQL = """
-- =========================================================
-- COUCHE 1 : STAGING (bronze) - donnees brutes telles quelles
-- =========================================================

CREATE TABLE IF NOT EXISTS staging_taches_2024 (
    id_tache                VARCHAR(20),
    matricule_agent         VARCHAR(50),
    type_contrat            VARCHAR(30),
    service                 VARCHAR(50),
    type_processus          VARCHAR(100),
    date_creation           VARCHAR(30),
    date_prise_en_charge    VARCHAR(30),
    date_cloture            VARCHAR(30),
    temps_passe_declare_min VARCHAR(20),
    volume_dossiers         VARCHAR(20),
    statut                  VARCHAR(30),
    complexite              VARCHAR(20),
    commentaire             TEXT,
    source_fichier          VARCHAR(50) DEFAULT 'extractions_taches_2024.csv'
);

CREATE TABLE IF NOT EXISTS staging_taches_2025 (
    id_tache                VARCHAR(20),
    matricule_agent         VARCHAR(50),
    type_contrat            VARCHAR(30),
    service                 VARCHAR(50),
    type_processus          VARCHAR(100),
    date_creation           VARCHAR(30),
    date_prise_en_charge    VARCHAR(30),
    date_cloture            VARCHAR(30),
    temps_passe_declare_min VARCHAR(20),
    volume_dossiers         VARCHAR(20),
    statut                  VARCHAR(30),
    complexite              VARCHAR(20),
    commentaire             TEXT,
    source_fichier          VARCHAR(50) DEFAULT 'extractions_taches_2025.xlsx'
);

CREATE TABLE IF NOT EXISTS staging_agents (
    matricule               VARCHAR(50),
    service                 VARCHAR(50),
    equipe                  VARCHAR(50),
    poste                   VARCHAR(100),
    type_contrat            VARCHAR(30),
    temps_travail           VARCHAR(30),
    date_entree             VARCHAR(30),
    date_fin_contrat        VARCHAR(30),
    actif                   VARCHAR(5),
    agence_interim          VARCHAR(100),
    ecole                   VARCHAR(100),
    societe_prestataire     VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS staging_absences (
    matricule               VARCHAR(50),
    type_contrat            VARCHAR(30),
    service                 VARCHAR(50),
    type_absence            VARCHAR(50),
    date_debut              VARCHAR(30),
    date_fin                VARCHAR(30),
    duree_jours             VARCHAR(10),
    valide                  VARCHAR(5),
    commentaire_rh          TEXT
);

-- =========================================================
-- COUCHE 2 : CLEAN (silver) - donnees nettoyees et fusionnees
-- =========================================================

-- taches_clean = resultat du merge taches + agents (Bloc 8 du notebook
-- data_cleaning), une ligne par tache, dates typees, categories normalisees
CREATE TABLE IF NOT EXISTS taches_clean (
    id_tache                VARCHAR(20) PRIMARY KEY,
    matricule_agent         VARCHAR(50),
    type_contrat            VARCHAR(30),
    service                 VARCHAR(50),
    type_processus          VARCHAR(100),
    date_creation           DATE,
    date_prise_en_charge    DATE,
    date_cloture            DATE,
    temps_passe_declare_min NUMERIC,
    volume_dossiers         NUMERIC,
    statut                  VARCHAR(30),
    complexite              VARCHAR(30),
    commentaire             TEXT,
    source_fichier          VARCHAR(50),
    service_agent           VARCHAR(50),
    type_contrat_agent      VARCHAR(30),
    temps_travail           VARCHAR(30),
    date_entree             DATE,
    date_fin_contrat        DATE,
    actif                   VARCHAR(5)
);

-- agents_clean = referentiel agents dedoublonne et nettoye
CREATE TABLE IF NOT EXISTS agents_clean (
    matricule               VARCHAR(50) PRIMARY KEY,
    service                 VARCHAR(50),
    equipe                  VARCHAR(50),
    poste                   VARCHAR(100),
    type_contrat            VARCHAR(30),
    temps_travail           VARCHAR(30),
    date_entree             DATE,
    date_fin_contrat        DATE,
    actif                   VARCHAR(5),
    agence_interim          VARCHAR(100),
    ecole                   VARCHAR(100),
    societe_prestataire     VARCHAR(100)
);

-- absences_clean = absences nettoyees, dates typees, durees plausibles
CREATE TABLE IF NOT EXISTS absences_clean (
    matricule               VARCHAR(50),
    type_contrat            VARCHAR(30),
    service                 VARCHAR(50),
    type_absence            VARCHAR(50),
    date_debut              DATE,
    date_fin                DATE,
    duree_jours             NUMERIC,
    valide                  VARCHAR(5),
    commentaire_rh          TEXT
);

-- =========================================================
-- COUCHE 3 : MANAGER (saisie manuelle) - incidents ETP et
-- annonces d'activite saisis par le manager depuis la page
-- /incidents de l'app. Sert de regresseur exogene aux Blocs 3A/3B
-- (cf. modelisation_common.ajouter_features_incidents).
--
-- Schema commun aux deux types d'evenement (type_evenement =
-- 'incident_etp' ou 'annonce') :
--   - date_evenement = date de l'incident, ou date de DEBUT pour une
--     annonce (la date de fin saisie dans le formulaire est convertie
--     en duree_jours a l'ecriture, pas stockee telle quelle).
--   - duree_jours = duree estimee (incident) ou duree de la periode
--     annoncee (date_fin - date_debut + 1).
--   - nb_etp_impactes : rempli seulement pour un incident_etp.
--   - volume_supplementaire_estime : rempli seulement pour une annonce.
--   - motif : sous-categorie (ex. "absence imprevue", "campagne
--     commerciale"), issue du dropdown "type d'incident" / "type
--     d'evenement" du formulaire.
-- =========================================================
CREATE TABLE IF NOT EXISTS incidents_manager (
    id                              SERIAL PRIMARY KEY,
    date_evenement                  DATE NOT NULL,
    service                         VARCHAR(50) NOT NULL,
    type_evenement                  VARCHAR(20) NOT NULL,
    motif                           VARCHAR(50),
    nb_etp_impactes                 NUMERIC,
    duree_jours                     NUMERIC NOT NULL DEFAULT 1,
    volume_supplementaire_estime    NUMERIC,
    commentaire                     TEXT,
    date_saisie                     TIMESTAMP NOT NULL DEFAULT NOW()
);
"""

if __name__ == "__main__":
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLES_SQL))
    print("Tables staging + clean creees (ou deja existantes).")
    print("Les tables features_* seront creees automatiquement par feature_engineering.py")
