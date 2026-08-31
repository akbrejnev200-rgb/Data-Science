# ============================================================
# GENERATION DATASET SIMULE -- FICHIER 1 : AGENTS / EQUIPES
# 4 services x 3 equipes = 12 equipes, ~90 agents
# ============================================================

import json
import random
import hashlib
from datetime import date, timedelta

random.seed(42)

DATE_DEBUT = date(2024, 1, 1)
DATE_FIN   = date(2026, 6, 12)

SERVICES = ["Recouvrement", "Conformité KYC", "Crédit Immobilier", "Épargne & Placements"]
EQUIPES_PAR_SERVICE = ["A", "B", "C"]  # ex: Equipe A, equipe b, EQUIPE C (variations de casse volontaires)

# Variations de casse realistes pour le champ "equipe"
def nom_equipe(lettre):
    variantes = [f"Equipe {lettre}", f"equipe {lettre.lower()}", f"EQUIPE {lettre}", f"Eq. {lettre}"]
    return random.choice(variantes)

POSTES = {
    "Superviseur": 0.10,
    "Gestionnaire senior": 0.25,
    "Gestionnaire": 0.35,
    "Analyste": 0.15,
    "Charge de conformite": 0.15,
}

TYPES_CONTRAT = {
    "CDI": 0.55,
    "CDD": 0.15,
    "Intérimaire": 0.10,
    "Alternant": 0.10,
    "Prestataire": 0.10,
}

# Variantes de casse pour types de contrat (saleté volontaire)
CASSE_CONTRAT = {
    "CDI": ["CDI", "cdi"],
    "CDD": ["CDD", "cdd"],
    "Intérimaire": ["Intérimaire", "Interimaire", "INTERIMAIRE"],
    "Alternant": ["Alternant", "alternant"],
    "Prestataire": ["Prestataire", "PRESTATAIRE"],
}

CASSE_POSTE = {
    "Superviseur": ["Superviseur", "SUPERVISEUR"],
    "Gestionnaire senior": ["Gestionnaire senior", "gestionnaire senior", "GESTIONNAIRE SENIOR"],
    "Gestionnaire": ["Gestionnaire", "gestionnaire"],
    "Analyste": ["Analyste", "analyste"],
    "Charge de conformite": ["CHARGE DE CONFORMITE", "Charge de Conformite", "charge de conformite"],
}

AGENCES_INTERIM = ["Randstad", "Manpower", "Adecco", "Synergie"]
ECOLES = ["EFREI", "ESILV", "CFA Banque", "IUT Informatique", "ESG Finance"]
PRESTATAIRES = ["Sopra Steria", "Capgemini", "Atos", "CGI"]


def matricule():
    return hashlib.md5(str(random.random()).encode()).hexdigest()[:12]


def tirage_pondere(d):
    items, poids = zip(*d.items())
    return random.choices(items, weights=poids, k=1)[0]


def date_aleatoire(debut, fin):
    delta = (fin - debut).days
    return debut + timedelta(days=random.randint(0, delta))


def format_date_alea(d):
    """Voir commentaire détaillé dans 02_generate_taches_full.py :
    on évite les formats ambigus (%m/%d/%Y, %d/%m/%y) quand day<=12."""
    if d.day <= 12:
        formats = ["%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y"]
    else:
        formats = ["%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%y"]
    return d.strftime(random.choice(formats))


def generer_agent():
    mat = matricule()
    service = random.choice(SERVICES)
    lettre_eq = random.choice(EQUIPES_PAR_SERVICE)
    poste_cle = tirage_pondere(POSTES)
    poste = random.choice(CASSE_POSTE[poste_cle])
    contrat_cle = tirage_pondere(TYPES_CONTRAT)
    contrat = random.choice(CASSE_CONTRAT[contrat_cle])

    # Temps de travail
    if contrat_cle == "Alternant":
        temps_travail = "100%"
    elif random.random() < 0.08:
        temps_travail = random.choice(["80%", "0.8", "Temps partiel"])
    else:
        temps_travail = random.choice(["100%", "1.0", "Temps plein"])

    # Date d'entrée
    date_entree = date_aleatoire(date(2015, 1, 1), DATE_FIN - timedelta(days=30))

    # Date de fin de contrat (CDD, Intérimaire, Alternant, Prestataire ont souvent une fin)
    date_fin_contrat = None
    actif = True
    if contrat_cle in ("CDD", "Intérimaire", "Alternant"):
        # On force le contrat à chevaucher au moins partiellement la période
        # couverte par les données de tâches (01/01/2024 -> 12/06/2026),
        # sinon l'agent n'apparaîtrait jamais dans les tâches générées.
        PERIODE_DEBUT = date(2024, 1, 1)
        duree = random.randint(90, 730)

        # date_entree doit être <= DATE_FIN, et date_entree + duree doit être >= PERIODE_DEBUT
        # On choisit date_entree dans [PERIODE_DEBUT - duree, DATE_FIN - 30j] borné par 2015
        borne_min = max(date(2015, 1, 1), PERIODE_DEBUT - timedelta(days=duree))
        borne_max = DATE_FIN - timedelta(days=30)
        if borne_min > borne_max:
            borne_min = borne_max
        date_entree = date_aleatoire(borne_min, borne_max)

        date_fin = date_entree + timedelta(days=duree)
        date_fin_contrat = date_fin
        if date_fin < DATE_FIN and random.random() < 0.5:
            actif = False

    # Champs spécifiques selon contrat
    agence_interim = random.choice(AGENCES_INTERIM) if contrat_cle == "Intérimaire" else None
    ecole = random.choice(ECOLES) if contrat_cle == "Alternant" else None
    societe_prestataire = random.choice(PRESTATAIRES) if contrat_cle == "Prestataire" else None

    # Saleté : valeurs "actif" hétérogènes (bool, str, None)
    if random.random() < 0.05:
        actif_val = None
    elif random.random() < 0.05:
        actif_val = "N" if not actif else "Y"
    else:
        actif_val = actif

    # Saleté : ~3% sans service renseigné
    if random.random() < 0.03:
        service_out = None
    else:
        service_out = service

    return {
        "matricule": mat,
        "service": service_out,
        "equipe": nom_equipe(lettre_eq),
        "poste": poste,
        "type_contrat": contrat,
        "temps_travail": temps_travail,
        "date_entree": format_date_alea(date_entree),
        "date_fin_contrat": format_date_alea(date_fin_contrat) if date_fin_contrat else None,
        "actif": actif_val,
        "agence_interim": agence_interim,
        "ecole": ecole,
        "societe_prestataire": societe_prestataire,
        # méta cachée utilisée par les générateurs suivants (pas dans le JSON final réel,
        # mais on la garde dans un fichier séparé pour la cohérence inter-fichiers)
        "_service_reel": service,
        "_equipe_lettre": lettre_eq,
        "_contrat_cle": contrat_cle,
        "_date_entree_reelle": date_entree.isoformat(),
        "_date_fin_reelle": date_fin_contrat.isoformat() if date_fin_contrat else None,
        "_actif_reel": actif,
    }


def main(n_agents=90):
    agents = [generer_agent() for _ in range(n_agents)]

    # Quelques doublons volontaires (mêmes matricules, lignes dupliquées)
    nb_doublons = max(1, n_agents // 30)
    for _ in range(nb_doublons):
        agents.append(random.choice(agents[:n_agents]).copy())

    # Export "réel" (sans les champs _meta) -> agents_equipes.json
    agents_public = []
    for a in agents:
        a_pub = {k: v for k, v in a.items() if not k.startswith("_")}
        agents_public.append(a_pub)

    with open("agents_equipes.json", "w", encoding="utf-8") as f:
        json.dump(agents_public, f, ensure_ascii=False, indent=2)

    # Export "méta" interne pour les générateurs suivants (tâches, absences)
    # -> ne sera pas livré comme fichier source, juste un pivot technique
    with open("_agents_meta.json", "w", encoding="utf-8") as f:
        json.dump(agents, f, ensure_ascii=False, indent=2)

    print(f"✓ {len(agents_public)} lignes générées -> agents_equipes.json")
    print(f"   ({nb_doublons} doublons inclus volontairement)")

    # Petit récap répartition
    from collections import Counter
    rep_service = Counter(a["_service_reel"] for a in agents[:n_agents])
    rep_contrat = Counter(a["_contrat_cle"] for a in agents[:n_agents])
    print("\nRépartition par service :", dict(rep_service))
    print("Répartition par contrat :", dict(rep_contrat))


if __name__ == "__main__":
    main()
