"""
theme.py

Jetons de couleur de l'application, alignes sur la maquette visuelle
validee (sidebar institutionnelle sombre, palette neutre chaude, rouge SG
reserve aux alertes). Utilises a la fois par le CSS (assets/custom.css,
memes valeurs recopiees car Dash ne partage pas nativement les variables
Python <-> CSS) et par les figures Plotly.
"""

SG_ROUGE = "#e60028"

INK = "#14130f"
INK_SOFT = "#3a372f"
MUTED = "#6b6759"
FAINT = "#948f80"
PAPER = "#f6f4ee"
SURFACE = "#ffffff"
LIGNE = "#e5e1d5"
LIGNE_DOUCE = "#efece2"

PANEL = "#1c1a16"
PANEL_LIGNE = "#35322a"
PANEL_ENCRE = "#f2f0e8"
PANEL_ATTENUE = "#a8a396"

BON = "#1a7a3c"
BON_FOND = "#e9f5ec"
AVERTISSEMENT = "#a3660a"
AVERTISSEMENT_FOND = "#faf0dd"

# Couleurs categorielles (identite des 4 services), ordre fixe
SERVICES_COULEURS = {
    "Crédit Immobilier": "#2a78d6",
    "Conformité KYC": "#c66a3a",
    "Recouvrement": "#8a8570",
    "Épargne & Placements": "#5c8a6e",
}
ORDRE_SERVICES = list(SERVICES_COULEURS.keys())

SEQUENTIEL_BLEU = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]

POLICE = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
