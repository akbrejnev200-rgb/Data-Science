# ============================================================
# app.py — Point d'entrée de l'application Dash
# Système de pilotage prédictif — Société Générale
# Filière Opérations Personnes Physiques
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dash import Dash, html, dcc, Input, Output

from auth import init_auth
from data import loader
from data.loader import charger_donnees, charger_modeles, etendre_djs_avec_previsions, TODAY, NOM_JOURS, NOM_MOIS

# — Initialisation de l'application —
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="Pilotage Prédictif SG",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# L'authentification doit être initialisée avant tout chargement de donnees
# lourd : si .env est mal configure (cles manquantes), l'app echoue vite
# plutot qu'apres un chargement SQL de plusieurs secondes.
init_auth(server)

# — Chargement unique des données et modèles (côté serveur, au démarrage) —
print("Chargement des données...")
dj, djs, df_feat, df_anomalies, df_previsions = charger_donnees()
df_charge_pred = loader.charger_predictions_charge()
df_retard_conf = loader.charger_confusion_retard()
taux_anomalies_global = loader.charger_taux_anomalies_global()
df_agents, df_absences = loader.charger_agents_absences()
taux_absence_repli = loader.calculer_taux_absence_repli(df_agents, df_absences)
print("Chargement des modèles...")
m_volumes, m_charge, m_retard, m_anomaly = charger_modeles()
print("Données et modèles chargés.")
djs = etendre_djs_avec_previsions(djs, df_previsions)

df_taches_en_cours = loader.charger_taches_en_cours()
df_risque_retard = loader.scorer_risque_retard(m_retard, df_taches_en_cours)

# — Import des pages et callbacks —
import pages.surcharge as pg_surcharge
import pages.reallocation as pg_reallocation
import pages.derive as pg_derive
import pages.fiabilite as pg_fiabilite
import pages.incidents as pg_incidents

import callbacks.cb_surcharge as cb_surcharge
import callbacks.cb_reallocation as cb_reallocation
import callbacks.cb_derive as cb_derive
import callbacks.cb_fiabilite as cb_fiabilite
import callbacks.cb_incidents as cb_incidents

from components.layout_common import sidebar, topbar

SOUS_TITRES = {
    "surcharge": f"Dernière donnée réelle · {TODAY.day} {NOM_MOIS[TODAY.month]} {TODAY.year}",
    "reallocation": "Comparaison de charge entre services",
    "derive": "Détection d'anomalies au niveau agent",
    "fiabilite": "Performance mesurée des modèles",
    "incidents": "Incidents ETP et annonces d'activité",
}
TITRES = {"surcharge": "Surcharge & prévisions", "reallocation": "Réallocation équitable",
          "derive": "Dérive individuelle", "fiabilite": "Fiabilité du modèle",
          "incidents": "Incidents & Annonces"}


# — Layout principal —
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="sidebar-container"),
    html.Div([
        html.Div(id="topbar-container"),
        html.Div(id="page-content"),
    ], className="main"),
], className="app-shell")


# — Routing —
@app.callback(
    Output("page-content", "children"),
    Output("sidebar-container", "children"),
    Output("topbar-container", "children"),
    Input("url", "pathname"),
)
def router(pathname):
    cle = pathname.strip("/") or "surcharge"
    if cle not in TITRES:
        cle = None

    if pathname in ("/", "/surcharge"):
        contenu = pg_surcharge.layout(dj, djs, df_previsions, m_charge, df_agents, df_absences, taux_absence_repli,
                                       df_risque_retard)
    elif pathname == "/reallocation":
        contenu = pg_reallocation.layout(djs, df_previsions, df_agents, df_absences, taux_absence_repli)
    elif pathname == "/derive":
        contenu = pg_derive.layout(df_feat, df_anomalies)
    elif pathname == "/fiabilite":
        contenu = pg_fiabilite.layout(dj, df_previsions, df_charge_pred, df_retard_conf,
                                       m_volumes, m_charge, m_retard, m_anomaly, taux_anomalies_global)
    elif pathname == "/incidents":
        contenu = pg_incidents.layout()
    else:
        contenu = html.Div([
            html.H2("Page introuvable", style={"color": "#e60028", "padding": "32px"}),
            dcc.Link("← Retour à l'accueil", href="/surcharge"),
        ])

    barre = topbar(TITRES.get(cle, "Pilotage Prédictif"), SOUS_TITRES.get(cle, ""))
    return contenu, sidebar(pathname), barre


# — Enregistrement des callbacks —
cb_surcharge.register(app, dj, djs, df_previsions, m_charge, df_agents, df_absences, taux_absence_repli,
                       df_risque_retard)
cb_reallocation.register(app, djs, df_previsions, df_agents, df_absences, taux_absence_repli)
cb_derive.register(app, df_feat, df_anomalies)
cb_fiabilite.register(app, dj, df_previsions, df_charge_pred, df_retard_conf,
                       m_volumes, m_charge, m_retard, m_anomaly)
cb_incidents.register(app)


# — Lancement —
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
