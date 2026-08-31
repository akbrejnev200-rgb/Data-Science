# ============================================================
# GENERATION DATASET SIMULE -- FICHIER 2 : TACHES (complet)
# Split : CSV 2024 (format colonnes "2024") + XLSX 2025-2026 (format "2025")
# ============================================================

import json
import random
import pandas as pd
import numpy as np
from datetime import date, timedelta

random.seed(43)
np.random.seed(43)

DATE_DEBUT = date(2024, 1, 1)
DATE_FIN_2024 = date(2024, 12, 31)
DATE_DEBUT_2025 = date(2025, 1, 1)
DATE_FIN = date(2026, 6, 12)

PROCESSUS_PAR_SERVICE = {
    "Recouvrement": [
        ("Relance amiable",              "Simple",   5, 15),
        ("Relance téléphonique",         "Moyen",    15, 30),
        ("Mise en demeure",              "Complexe", 45, 90),
        ("Plan d'apurement",             "Complexe", 60, 120),
        ("Archivage numérique",          "Simple",   5, 12),
        ("Contrôle pièces justificatives","Moyen",   15, 35),
    ],
    "Conformité KYC": [
        ("Contrôle pièces justificatives","Moyen",   15, 35),
        ("Vérification identité",        "Moyen",    20, 40),
        ("Vérification PEP",             "Complexe", 45, 100),
        ("Mise à jour dossier client",   "Simple",   8, 20),
        ("Revue périodique KYC",         "Complexe", 50, 110),
        ("Archivage numérique",          "Simple",   5, 12),
    ],
    "Crédit Immobilier": [
        ("Étude de dossier",             "Complexe", 45, 120),
        ("Vérification pièces revenus",  "Moyen",    20, 45),
        ("Édition offre de prêt",        "Moyen",    15, 30),
        ("Déblocage de fonds",           "Complexe", 30, 75),
        ("Archivage numérique",          "Simple",   5, 12),
        ("Contrôle pièces justificatives","Moyen",   15, 35),
    ],
    "Épargne & Placements": [
        ("Ouverture de produit",         "Moyen",    15, 35),
        ("Arbitrage de portefeuille",    "Complexe", 30, 70),
        ("Clôture de produit",           "Moyen",    15, 30),
        ("Mise à jour dossier client",   "Simple",   8, 20),
        ("Contrôle pièces justificatives","Moyen",   15, 35),
        ("Archivage numérique",          "Simple",   5, 12),
    ],
}

STATUTS = {"Clôturé": 0.78, "En cours": 0.12, "Suspendu": 0.07, "Inconnu": 0.03}
CASSE_STATUT = {
    "Clôturé": ["CLOTURE", "cloture", "Clôturé", "Fermé"],
    "En cours": ["EN_COURS", "En cours"],
    "Suspendu": ["Suspendu"],
    "Inconnu": ["Inconnu", "", None],
}
CASSE_COMPLEXITE = {
    "Simple": ["Simple", "simple"],
    "Moyen": ["Moyen", "moyen"],
    "Complexe": ["Complexe", "COMPLEXE"],
}
CASSE_CONTRAT_TACHE = {
    "CDI": ["CDI", "cdi"],
    "CDD": ["CDD", "cdd"],
    "Intérimaire": ["Intérimaire", "Interimaire", "INTERIMAIRE"],
    "Alternant": ["Alternant", "alternant"],
    "Prestataire": ["Prestataire", "PRESTATAIRE"],
}
COEFF_CONTRAT = {"CDI": 1.00, "Prestataire": 0.90, "CDD": 0.85, "Intérimaire": 0.75, "Alternant": 0.60}
COEFF_POSTE = {"Superviseur": 1.10, "Gestionnaire senior": 1.15, "Gestionnaire": 1.00,
               "Analyste": 0.95, "Charge de conformite": 1.00}


def format_date_alea(d):
    """
    Formate une date avec un format pris au hasard parmi plusieurs (saleté
    volontaire de saisie). Pour les jours <= 12, les formats %m/%d/%Y et
    %d/%m/%y créent une ambiguïté jour/mois qui, une fois ré-interprétée par
    le nettoyage (priorité %d/%m/%Y), peut faire "sortir" la date de la
    période réelle couverte par le dataset (ex: 7 janvier -> '01/07/2026' ->
    relu comme 1er juillet). On exclut donc ces formats ambigus quand le jour
    est <= 12, pour ne garder que des formats non ambigus dans ce cas.
    """
    if d.day <= 12:
        formats = ["%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y"]
    else:
        formats = ["%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%y"]
    return d.strftime(random.choice(formats))


def tirage_pondere(d):
    items, poids = zip(*d.items())
    return random.choices(items, weights=poids, k=1)[0]


def get_poste_cle(poste_str):
    p = poste_str.lower()
    if "superviseur" in p: return "Superviseur"
    if "senior" in p: return "Gestionnaire senior"
    if "gestionnaire" in p: return "Gestionnaire"
    if "analyste" in p: return "Analyste"
    if "conformite" in p or "conformité" in p: return "Charge de conformite"
    return "Gestionnaire"


def jour_est_ouvre(d):
    return d.weekday() < 5


def agent_actif_a_date(agent, jour):
    """Vérifie si l'agent est actif (présent dans l'effectif) à cette date."""
    date_entree = date.fromisoformat(agent["_date_entree_reelle"])
    if jour < date_entree:
        return False
    if agent["_date_fin_reelle"]:
        date_fin = date.fromisoformat(agent["_date_fin_reelle"])
        if jour > date_fin:
            return False
    return True


_id_counter = [0]
def next_id():
    _id_counter[0] += 1
    return f"T{100000 + _id_counter[0]}"


def facteur_charge_jour(jour):
    """
    Calcule un facteur multiplicatif de charge pour une date donnée,
    combinant plusieurs effets réalistes :
    - tendance annuelle : croissance progressive de l'activité (+15% sur 2.5 ans)
    - saisonnalité fin de trimestre : pics en mars/juin/sept/déc (clôtures bancaires)
    - effet fin de mois : légère hausse les derniers jours du mois
    - creux estival : baisse en août (congés clients/agents)
    Ce facteur donne du signal exploitable par Prophet/SARIMA (tendance+saisonnalité)
    et par les features calendaires des modèles de régression.
    """
    # 1) Tendance annuelle linéaire : de 1.0 (01/01/2024) à ~1.18 (12/06/2026)
    nb_jours_total = (DATE_FIN - DATE_DEBUT).days
    progression = (jour - DATE_DEBUT).days / nb_jours_total
    facteur_tendance = 1.0 + 0.18 * progression

    # 2) Saisonnalité fin de trimestre (pic clôtures bancaires)
    facteur_trimestre = 1.0
    if jour.month in (3, 6, 9, 12):
        facteur_trimestre = 1.35
    elif jour.month in (1, 4, 7, 10):
        # mois suivant un trimestre : légère baisse (rattrapage terminé)
        facteur_trimestre = 0.85

    # 3) Effet fin de mois (derniers 5 jours ouvrés du mois)
    facteur_fin_mois = 1.25 if jour.day >= 25 else 1.0

    # 4) Creux estival (août)
    facteur_ete = 0.78 if jour.month == 8 else 1.0

    return facteur_tendance * facteur_trimestre * facteur_fin_mois * facteur_ete


def facteur_complexite_jour(jour):
    """
    Les jours de forte charge (fin de trimestre, fin de mois) ont une
    proportion plus élevée de dossiers complexes (moins de temps pour trier,
    traitement des dossiers en attente les plus anciens/complexes).
    Retourne un facteur de pondération additionnel pour le tirage Complexe.
    """
    bonus = 0.0
    if jour.month in (3, 6, 9, 12):
        bonus += 0.08
    if jour.day >= 25:
        bonus += 0.05
    return bonus


def generer_taches_pour_agent_jour(agent, jour):
    service = agent["_service_reel"] or random.choice(list(PROCESSUS_PAR_SERVICE.keys()))
    processus_dispo = PROCESSUS_PAR_SERVICE[service]
    contrat_cle = agent["_contrat_cle"]
    coeff_contrat = COEFF_CONTRAT.get(contrat_cle, 0.85)
    coeff_poste = COEFF_POSTE.get(get_poste_cle(agent["poste"]), 1.0)
    coeff_global = coeff_contrat * coeff_poste

    # Nombre de tâches de base (14-22), modulé par le facteur de charge du jour
    # (tendance annuelle + saisonnalité trimestre/fin de mois/été)
    facteur_jour = facteur_charge_jour(jour)
    nb_taches_base = 18  # fixe (était 14-22 puis 16-20) : tout le signal vient du facteur saisonnier
    nb_taches = max(1, round(nb_taches_base * facteur_jour))

    bonus_complexite = facteur_complexite_jour(jour)

    lignes = []
    for _ in range(nb_taches):
        # Tirage du processus : les jours de forte charge ont plus de
        # processus complexes (bonus_complexite ajouté au poids "Complexe")
        if bonus_complexite > 0 and random.random() < bonus_complexite:
            candidats_complexes = [p for p in processus_dispo if p[1] == "Complexe"]
            if candidats_complexes:
                nom_proc, complexite, t_min, t_max = random.choice(candidats_complexes)
            else:
                nom_proc, complexite, t_min, t_max = random.choice(processus_dispo)
        else:
            nom_proc, complexite, t_min, t_max = random.choice(processus_dispo)

        temps_reel = (random.uniform(t_min, t_max) / coeff_global) * random.uniform(0.92, 1.08)
        temps_reel = max(2, round(temps_reel))

        if complexite == "Simple":
            volume = random.choices([1, 2, 3, 5, 10], weights=[40, 25, 15, 12, 8])[0]
        elif complexite == "Moyen":
            volume = random.choices([1, 2, 3, 5], weights=[55, 25, 12, 8])[0]
        else:
            volume = random.choices([1, 2], weights=[85, 15])[0]

        temps_total = temps_reel * volume

        date_creation = jour
        delai_prise = random.choices([0, 1, 2, 3], weights=[55, 25, 12, 8])[0]
        date_prise = date_creation + timedelta(days=delai_prise)

        statut_cle = tirage_pondere(STATUTS)
        date_cloture = None
        if statut_cle == "Clôturé":
            # Durée de traitement corrélée à : la complexité de la tâche,
            # le coefficient global de l'agent (contrat+poste), et la charge
            # du jour (facteur_jour). Plus la tâche est complexe, l'agent
            # moins productif, et le jour chargé -> durée plus longue.
            # Cela donne un vrai signal exploitable pour la classification
            # du risque de retard (Bloc 3C).
            base_complexite = {"Simple": 1.0, "Moyen": 1.6, "Complexe": 2.6}[complexite]
            facteur_agent = 1.0 / coeff_global       # agent moins productif -> durée plus longue
            facteur_charge_duree = 0.6 + 0.5 * facteur_jour  # jour chargé -> durée plus longue

            duree_moyenne = base_complexite * facteur_agent * facteur_charge_duree
            duree_trait = max(1, round(random.gauss(duree_moyenne, 0.7)))
            date_cloture = date_prise + timedelta(days=duree_trait)

        if date_cloture and random.random() < 0.02:
            date_cloture = date_creation - timedelta(days=random.randint(1, 5))

        statut_out = random.choice(CASSE_STATUT[statut_cle])
        complexite_out = random.choice(CASSE_COMPLEXITE[complexite])

        if random.random() < 0.015:
            volume = volume * random.choice([100, 1000])

        temps_out = temps_total
        if random.random() < 0.01:
            temps_out = -abs(temps_total)
        if random.random() < 0.03:
            temps_out = None

        commentaire = random.choice([None, "", "RAS", "OK", "A vérifier", "Traité", "URGENT"])

        lignes.append({
            "ID_Tache": next_id(),
            "Matricule_Agent": agent["matricule"],
            "Type_Contrat": random.choice(CASSE_CONTRAT_TACHE.get(contrat_cle, [contrat_cle])),
            "Service": service,
            "Type_Processus": nom_proc,
            "Date_Creation": date_creation,
            "Date_Prise_En_Charge": date_prise,
            "Date_Cloture": date_cloture,
            "Temps_Passe_Declare_Min": temps_out,
            "Volume_Dossiers": volume,
            "Statut": statut_out,
            "Complexite": complexite_out,
            "Commentaire": commentaire,
        })
    return lignes


def generer_periode(agents, date_debut, date_fin):
    jour = date_debut
    toutes_lignes = []
    while jour <= date_fin:
        if jour_est_ouvre(jour):
            for agent in agents:
                if agent_actif_a_date(agent, jour):
                    toutes_lignes.extend(generer_taches_pour_agent_jour(agent, jour))
        jour += timedelta(days=1)
    return toutes_lignes


def main():
    with open("_agents_meta.json", "r", encoding="utf-8") as f:
        agents = json.load(f)

    print("Génération 2024...")
    lignes_2024 = generer_periode(agents, DATE_DEBUT, DATE_FIN_2024)
    print(f"  -> {len(lignes_2024):,} lignes")

    print("Génération 2025-2026...")
    lignes_2025 = generer_periode(agents, DATE_DEBUT_2025, DATE_FIN)
    print(f"  -> {len(lignes_2025):,} lignes")

    # ---------- Export CSV 2024 (format colonnes "2024") ----------
    df_2024 = pd.DataFrame(lignes_2024)
    for col in ["Date_Creation", "Date_Prise_En_Charge", "Date_Cloture"]:
        df_2024[col] = df_2024[col].apply(lambda d: format_date_alea(d) if pd.notna(d) else None)
    df_2024.to_csv("extractions_taches_2024.csv", index=False, encoding="utf-8")
    print(f"✓ extractions_taches_2024.csv exporté -> {len(df_2024):,} lignes")

    # ---------- Export XLSX 2025-2026 (format colonnes "2025") ----------
    rename_2025 = {
        "ID_Tache": "id tache",
        "Matricule_Agent": "MATRICULE AGENT",
        "Type_Contrat": "type contrat",
        "Service": "Service ",  # espace en trop, comme original
        "Type_Processus": "type_processus",
        "Date_Creation": "date creation",
        "Date_Prise_En_Charge": "Date Prise En Charge",
        "Date_Cloture": "DATE_CLOTURE",
        "Temps_Passe_Declare_Min": "tps_passe (min)",
        "Volume_Dossiers": "Nb Dossiers",
        "Statut": "STATUT",
        "Complexite": "complexite",
        "Commentaire": "commentaires",
    }
    df_2025 = pd.DataFrame(lignes_2025)
    for col in ["Date_Creation", "Date_Prise_En_Charge", "Date_Cloture"]:
        df_2025[col] = df_2025[col].apply(lambda d: format_date_alea(d) if pd.notna(d) else None)
    df_2025 = df_2025.rename(columns=rename_2025)

    # Insérer une ligne vide au milieu (saleté volontaire, comme l'original)
    milieu = len(df_2025) // 2
    df_top = df_2025.iloc[:milieu]
    df_bottom = df_2025.iloc[milieu:]
    ligne_vide = pd.DataFrame([{c: np.nan for c in df_2025.columns}])
    df_2025_final = pd.concat([df_top, ligne_vide, df_bottom], ignore_index=True)

    # Onglet parasite "Récap_Auto"
    df_recap = pd.DataFrame({
        "Indicateur": ["Total tâches", "Date export", "Auteur"],
        "Valeur": [len(df_2025), str(DATE_FIN), "Auto-généré"]
    })

    with pd.ExcelWriter("extractions_taches_2025.xlsx", engine="openpyxl") as writer:
        df_2025_final.to_excel(writer, sheet_name="Taches_2025", index=False)
        df_recap.to_excel(writer, sheet_name="Récap_Auto", index=False)

    print(f"✓ extractions_taches_2025.xlsx exporté -> {len(df_2025_final):,} lignes (+ 1 ligne vide + onglet parasite)")

    print(f"\nTOTAL BRUT : {len(df_2024) + len(df_2025_final):,} lignes")


if __name__ == "__main__":
    main()
