# ============================================================
# GENERATION DATASET SIMULE -- FICHIER 3 : ABSENCES / CONGES
# ~3000 lignes, coherent avec agents_equipes et la periode
# ============================================================

import json
import random
import pandas as pd
from datetime import date, timedelta

random.seed(44)

DATE_DEBUT = date(2024, 1, 1)
DATE_FIN = date(2026, 6, 12)

TYPES_ABSENCE = {
    "CP": 0.35,
    "RTT": 0.25,
    "Maladie": 0.20,
    "Formation": 0.13,
    "Semaine école": 0.04,  # surtout alternants
    "Absent": 0.03,
}

CASSE_ABSENCE = {
    "CP": ["CP", "Congé Payé", "CONGE_PAYE"],
    "RTT": ["RTT", "rtt"],
    "Maladie": ["Maladie", "MALADIE"],
    "Formation": ["Formation", "FORMATION", "Formation CFA"],
    "Semaine école": ["Semaine école", "SEMAINE_ECOLE", "Formation CFA"],
    "Absent": ["Absent"],
}

CASSE_CONTRAT = {
    "CDI": ["CDI", "cdi"],
    "CDD": ["CDD", "cdd"],
    "Intérimaire": ["Intérimaire", "Interimaire", "INTERIMAIRE"],
    "Alternant": ["Alternant", "alternant"],
    "Prestataire": ["Prestataire", "PRESTATAIRE"],
}

VALIDE_VARIANTES = ["O", "OUI", "N", "NON", "0", "1", "Y"]
COMMENTAIRES_RH = [None, "", "RAS", "Certificat fourni", "En attente validation",
                   "A régulariser", "URGENT", "Validé manager"]


def format_date_alea(d):
    """Voir commentaire détaillé dans 02_generate_taches_full.py :
    on évite les formats ambigus (%m/%d/%Y, %d/%m/%y) quand day<=12."""
    if d.day <= 12:
        formats = ["%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y"]
    else:
        formats = ["%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%y"]
    return d.strftime(random.choice(formats))


def tirage_pondere(d):
    items, poids = zip(*d.items())
    return random.choices(items, weights=poids, k=1)[0]


def date_aleatoire(debut, fin):
    delta = (fin - debut).days
    if delta <= 0:
        return debut
    return debut + timedelta(days=random.randint(0, delta))


def date_aleatoire_biaisee_fin_periode(debut, fin, proba_biais=0.55):
    """
    Tire une date aléatoire avec une probabilité accrue de tomber en
    fin de mois ou fin de trimestre. Modélise un effet réaliste de
    "mauvaise synchronisation" : les agents posent souvent leurs
    RTT/formations en fin de mois (moment qui leur semble calme côté
    dossiers personnels), ce qui réduit l'ETP disponible justement
    quand la charge entrante est au plus haut (fin de mois/trimestre).
    C'est précisément ce type de déséquilibre que le pilier 2 du projet
    (allocation équitable des ressources) vise à corriger.
    """
    if random.random() < proba_biais:
        # On tire un mois/jour dans la plage, puis on le pousse vers la fin du mois
        d = date_aleatoire(debut, fin)
        jour_cible = random.randint(24, 28)
        try:
            d = d.replace(day=jour_cible)
        except ValueError:
            pass  # mois plus court, on garde la date telle quelle
        if d < debut:
            d = debut
        if d > fin:
            d = fin
        return d
    return date_aleatoire(debut, fin)


def generer_absences_agent(agent, nb_absences):
    lignes = []
    date_entree = date.fromisoformat(agent["_date_entree_reelle"])
    date_fin_contrat = (date.fromisoformat(agent["_date_fin_reelle"])
                         if agent["_date_fin_reelle"] else DATE_FIN)

    borne_debut = max(DATE_DEBUT, date_entree)
    borne_fin = min(DATE_FIN, date_fin_contrat)
    if borne_debut >= borne_fin:
        return lignes

    contrat_cle = agent["_contrat_cle"]

    for _ in range(nb_absences):
        type_cle = tirage_pondere(TYPES_ABSENCE)
        # Semaine école surtout pour alternants
        if type_cle == "Semaine école" and contrat_cle != "Alternant":
            type_cle = "Formation"

        if type_cle in ("RTT", "Formation", "Semaine école"):
            date_debut_abs = date_aleatoire_biaisee_fin_periode(borne_debut, borne_fin)
        else:
            date_debut_abs = date_aleatoire(borne_debut, borne_fin)

        if type_cle == "CP":
            duree = random.randint(5, 10)
        elif type_cle == "RTT":
            duree = 1
        elif type_cle == "Maladie":
            duree = random.randint(1, 5)
        elif type_cle in ("Formation", "Semaine école"):
            duree = random.randint(2, 5)
        else:
            duree = 1

        date_fin_abs = date_debut_abs + timedelta(days=duree - 1)

        # ~2% durées aberrantes (saisie erronée)
        duree_out = duree
        if random.random() < 0.02:
            duree_out = random.choice([-1, 95, 120])

        # ~2% dates incohérentes (fin avant début)
        if random.random() < 0.02:
            date_fin_abs = date_debut_abs - timedelta(days=random.randint(1, 5))

        contrat_out = random.choice(CASSE_CONTRAT.get(contrat_cle, [contrat_cle]))
        type_out = random.choice(CASSE_ABSENCE[type_cle])
        valide_out = random.choice(VALIDE_VARIANTES)

        # ~3% service manquant
        service_out = agent["_service_reel"] if random.random() > 0.03 else None

        lignes.append({
            "Matricule": agent["matricule"],
            "Type_Contrat": contrat_out,
            "Service": service_out,
            "Type_Absence": type_out,
            "Date_Debut": format_date_alea(date_debut_abs),
            "Date_Fin": format_date_alea(date_fin_abs),
            "Duree_Jours": duree_out,
            "Valide": valide_out,
            "Commentaire_RH": random.choice(COMMENTAIRES_RH),
        })
    return lignes


def main(cible=None):
    with open("_agents_meta.json", "r", encoding="utf-8") as f:
        agents = json.load(f)

    agents_uniques = {a["matricule"]: a for a in agents}.values()
    agents_uniques = list(agents_uniques)

    # ~21 occurrences/an/agent pour viser ~30j absence/an/agent
    # sur 2.5 ans -> ~52 occurrences/agent
    nb_par_agent = 27

    toutes_lignes = []
    for agent in agents_uniques:
        nb = random.randint(max(1, nb_par_agent - 5), nb_par_agent + 5)
        toutes_lignes.extend(generer_absences_agent(agent, nb))

    # Quelques doublons volontaires
    nb_doublons = max(1, len(toutes_lignes) // 50)
    for _ in range(nb_doublons):
        toutes_lignes.append(random.choice(toutes_lignes).copy())

    df = pd.DataFrame(toutes_lignes)
    df.to_csv("absences_conges.csv", index=False, encoding="utf-8")
    print(f"✓ absences_conges.csv exporté -> {len(df):,} lignes ({nb_doublons} doublons inclus)")
    print(f"\nRépartition types absence :")
    print(df["Type_Absence"].value_counts())


if __name__ == "__main__":
    main()
