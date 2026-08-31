"""
pages/incidents.py

Page de saisie manuelle par le manager : incidents ETP (absence, maladie,
depart urgent...) et annonces d'activite (campagne, pic prevu...), ecrits
dans incidents_manager. Cette table alimente ensuite, en lecture seule
cote pipeline, les regresseurs exogenes nb_etp_impactes_jour /
volume_annonce_jour des Blocs 3A/3B (cf.
notebooks_avec_ecriture_dans_database/modelisation_common.py).

La table ne stocke qu'une date de debut (date_evenement) et une duree
(duree_jours) : pour une annonce, la date de fin saisie dans le formulaire
sert uniquement a calculer duree_jours a l'ecriture (cf.
callbacks/cb_incidents.py), jamais stockee telle quelle.

L'historique (30 derniers evenements) est recharge a chaque navigation
vers /incidents -- pas de cache, contrairement aux autres pages dont les
donnees sont chargees une fois au demarrage du serveur (app.py) : c'est la
seule page qui ecrit dans PostgreSQL, la fraicheur prime sur la performance.
"""

from datetime import date

import pandas as pd
from dash import dash_table, dcc, html

from components.layout_common import chart_block, page_wrapper
from components.theme import ORDRE_SERVICES
from data.loader import charger_incidents

OPTIONS_SERVICE = [{"label": s, "value": s} for s in ORDRE_SERVICES]

OPTIONS_MOTIF_INCIDENT = [
    {"label": "Absence imprévue", "value": "Absence imprévue"},
    {"label": "Arrêt maladie", "value": "Arrêt maladie"},
    {"label": "Départ urgent", "value": "Départ urgent"},
    {"label": "Autre", "value": "Autre"},
]

OPTIONS_MOTIF_ANNONCE = [
    {"label": "Campagne commerciale", "value": "Campagne commerciale"},
    {"label": "Opération exceptionnelle", "value": "Opération exceptionnelle"},
    {"label": "Pic prévu", "value": "Pic prévu"},
    {"label": "Autre", "value": "Autre"},
]

LABEL_TYPE_EVENEMENT = {"incident_etp": "Incident ETP", "annonce": "Annonce"}

# Champs a vider apres une saisie reussie, dans le meme ordre que les Output
# correspondants du callback de soumission (cf. cb_incidents.py).
CHAMPS_FORMULAIRE = [
    "incidents-date-etp", "incidents-service-etp", "incidents-nb-etp",
    "incidents-motif-etp", "incidents-duree-etp",
    "incidents-date-debut-annonce", "incidents-date-fin-annonce", "incidents-service-annonce",
    "incidents-motif-annonce", "incidents-volume-annonce", "incidents-commentaire-annonce",
]


def _champ(label, composant):
    return html.Div([html.Label(label, className="form-label"), composant], className="form-field")


def _bloc_formulaire_incident():
    return html.Div([
        _champ("Date", dcc.DatePickerSingle(
            id="incidents-date-etp", date=date.today().isoformat(),
            display_format="DD/MM/YYYY", className="form-datepicker",
        )),
        _champ("Service concerné", dcc.Dropdown(
            id="incidents-service-etp", options=OPTIONS_SERVICE,
            placeholder="Sélectionner un service", clearable=True,
        )),
        _champ("Nombre d'ETP impactés", dcc.Input(
            id="incidents-nb-etp", type="number", min=0, step=0.5,
            placeholder="Ex. 1.5", className="form-input",
        )),
        _champ("Type d'incident", dcc.Dropdown(
            id="incidents-motif-etp", options=OPTIONS_MOTIF_INCIDENT,
            placeholder="Sélectionner un type", clearable=True,
        )),
        _champ("Durée estimée (jours)", dcc.Input(
            id="incidents-duree-etp", type="number", min=1, step=1,
            placeholder="Ex. 5", className="form-input",
        )),
    ], id="incidents-form-etp", className="form-grid")


def _bloc_formulaire_annonce():
    return html.Div([
        _champ("Date de début", dcc.DatePickerSingle(
            id="incidents-date-debut-annonce", date=date.today().isoformat(),
            display_format="DD/MM/YYYY", className="form-datepicker",
        )),
        _champ("Date de fin", dcc.DatePickerSingle(
            id="incidents-date-fin-annonce", date=date.today().isoformat(),
            display_format="DD/MM/YYYY", className="form-datepicker",
        )),
        _champ("Service concerné", dcc.Dropdown(
            id="incidents-service-annonce", options=OPTIONS_SERVICE,
            placeholder="Sélectionner un service", clearable=True,
        )),
        _champ("Type d'événement", dcc.Dropdown(
            id="incidents-motif-annonce", options=OPTIONS_MOTIF_ANNONCE,
            placeholder="Sélectionner un type", clearable=True,
        )),
        _champ("Volume supplémentaire estimé (dossiers)", dcc.Input(
            id="incidents-volume-annonce", type="number", min=0, step=1,
            placeholder="Ex. 300", className="form-input",
        )),
        _champ("Commentaire", dcc.Textarea(
            id="incidents-commentaire-annonce", placeholder="Commentaire libre (optionnel)",
            className="form-textarea",
        )),
    ], id="incidents-form-annonce", className="form-grid", style={"display": "none"})


def construire_formulaire():
    return html.Div([
        dcc.Store(id="incidents-type-store", data="incident_etp"),
        html.Div([
            html.Button("Incident ETP", id="incidents-btn-etp", n_clicks=0,
                        className="type-toggle-btn type-toggle-btn-active"),
            html.Button("Annonce", id="incidents-btn-annonce", n_clicks=0,
                        className="type-toggle-btn"),
        ], className="type-toggle"),
        _bloc_formulaire_incident(),
        _bloc_formulaire_annonce(),
        html.Div([
            html.Button("Enregistrer l'événement", id="incidents-submit", n_clicks=0,
                        className="btn-accent"),
            html.Div(id="incidents-feedback"),
        ], className="form-actions"),
    ], className="incidents-form")


def construire_table_historique(df_hist):
    if not len(df_hist):
        return html.P("Aucun événement saisi pour le moment.", className="table-note")

    colonnes_affichees = ["Date", "Service", "Type", "Motif", "ETP impactés", "Durée (j)",
                           "Volume estimé", "Commentaire", "Saisi le"]

    lignes = []
    for _, r in df_hist.iterrows():
        lignes.append({
            "id": int(r["id"]),
            "Date": r["date_evenement"].strftime("%d/%m/%Y") if pd.notna(r["date_evenement"]) else "",
            "Service": r["service"],
            "Type": LABEL_TYPE_EVENEMENT.get(r["type_evenement"], r["type_evenement"]),
            "Motif": r["motif"] or "",
            "ETP impactés": "" if pd.isna(r["nb_etp_impactes"]) else f"{r['nb_etp_impactes']:g}",
            "Durée (j)": "" if pd.isna(r["duree_jours"]) else f"{r['duree_jours']:g}",
            "Volume estimé": "" if pd.isna(r["volume_supplementaire_estime"]) else f"{r['volume_supplementaire_estime']:.0f}",
            "Commentaire": r["commentaire"] or "",
            "Saisi le": r["date_saisie"].strftime("%d/%m/%Y %H:%M") if pd.notna(r["date_saisie"]) else "",
        })

    return dash_table.DataTable(
        id="incidents-history-table",
        data=lignes,
        columns=[{"name": c, "id": c} for c in colonnes_affichees],
        row_deletable=True,
        page_size=30,
        style_cell={"fontFamily": "-apple-system, sans-serif", "fontSize": "12.5px", "padding": "9px 12px",
                    "border": "none", "fontVariantNumeric": "tabular-nums", "textAlign": "left"},
        style_cell_conditional=[
            {"if": {"column_id": c}, "textAlign": "right"}
            for c in ["ETP impactés", "Durée (j)", "Volume estimé"]
        ] + [{"if": {"column_id": "Commentaire"}, "maxWidth": "240px", "whiteSpace": "normal"}],
        style_header={"fontWeight": "600", "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": "0.03em",
                      "backgroundColor": "#f6f4ee", "color": "#6b6759", "border": "none",
                      "borderBottom": "1px solid #e5e1d5"},
        style_data={"border": "none", "borderBottom": "1px solid #efece2", "color": "#3a372f"},
        style_as_list_view=True,
    )


def layout():
    df_hist = charger_incidents(30)
    contenu = [
        chart_block("Nouvel événement", construire_formulaire()),
        html.Div(id="incidents-history-block", children=chart_block(
            "Historique des 30 derniers événements",
            construire_table_historique(df_hist),
            "Cliquer sur la croix en début de ligne pour supprimer un événement (définitif).",
        )),
    ]
    return page_wrapper(*contenu)
