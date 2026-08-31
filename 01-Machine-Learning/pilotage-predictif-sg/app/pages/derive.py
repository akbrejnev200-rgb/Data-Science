"""
pages/derive.py

Detection d'anomalies au niveau tache/agent — signal pour un manager,
jamais une decision RH automatique.
"""

import plotly.graph_objects as go
from dash import dcc, html

from components.layout_common import chart_block, figure_layout_defaults, kpi_card, kpi_row, page_wrapper
from components.theme import ORDRE_SERVICES, SERVICES_COULEURS, SG_ROUGE

NOTE_ETHIQUE = "Signal à vérifier avec l'agent, jamais une évaluation automatisée."

# Le modele flague ~2% de TOUTES les taches (contamination=0.02) : avec des
# centaines/milliers de taches par agent, chaque agent a statistiquement au
# moins une tache flaguee -- un simple "a au moins 1 anomalie" ne veut donc
# rien dire (90/90 agents dans ce cas). Le vrai signal est le TAUX d'anomalie
# de l'agent compare au taux global, pas le compte brut (qui n'est qu'un
# proxy du nombre de taches traitees). Seuil : 3x le taux global.
MULTIPLICATEUR_SEUIL = 3


def _taux_par_agent(df_anomalies):
    par_agent = df_anomalies["par_agent"].groupby("Matricule_Agent").agg(
        nb_taches=("nb_taches_evaluees", "sum"), nb_anomalies=("nb_anomalies", "sum"),
    ).reset_index()
    par_agent["taux_pct"] = par_agent["nb_anomalies"] / par_agent["nb_taches"] * 100
    return par_agent


def construire_kpis(df_feat, df_anomalies):
    par_agent = _taux_par_agent(df_anomalies)
    n_agents = len(df_feat)
    taux_global = df_anomalies["par_agent"]["nb_anomalies"].sum() / df_anomalies["par_agent"]["nb_taches_evaluees"].sum() * 100
    seuil = taux_global * MULTIPLICATEUR_SEUIL
    n_taux_eleve = int((par_agent["taux_pct"] > seuil).sum())
    prod_moy = df_feat["productivite_moy"].mean()

    return kpi_row(
        kpi_card("Agents suivis", n_agents, "au total"),
        kpi_card("Agents à taux anormal", n_taux_eleve, f"> {seuil:.1f}% (3x le taux global {taux_global:.1f}%)",
                  statut="avertissement" if n_taux_eleve else "bon"),
        kpi_card("Productivité moyenne", f"{prod_moy:.1f}", "dossiers / heure"),
        kpi_card("Taux d'anomalie le plus élevé", f"{par_agent['taux_pct'].max():.1f}%", "d'un seul agent"),
    )


TOUS_SERVICES = "Tous services"


def construire_fig_scatter(df_feat, df_anomalies, service=TOUS_SERVICES):
    fusion = df_feat.merge(_taux_par_agent(df_anomalies), on="Matricule_Agent", how="left")
    fusion["taux_pct"] = fusion["taux_pct"].fillna(0)
    productivite_moyenne = fusion["productivite_moy"].mean()

    # Taille = TAUX d'anomalie (pas le compte brut) : un agent qui traite
    # plus de taches a mecaniquement plus d'anomalies en valeur absolue sans
    # etre plus atypique -- seul le taux (proportion) est comparable entre
    # agents. sizemode="area" + sizeref bornent le diametre rendu.
    DIAMETRE_MAX = 30
    DIAMETRE_MIN = 6
    sizeref = 2.0 * fusion["taux_pct"].max() / (DIAMETRE_MAX ** 2)

    fig = go.Figure()

    # Repere de lecture : sans ligne moyenne, un point isole n'a pas de sens
    # (au-dessus/en-dessous de quoi ?). Position "bottom right" pour ne pas
    # chevaucher la legende des services (haut-gauche).
    fig.add_hline(
        y=productivite_moyenne, line=dict(color="#c3c2b7", width=1, dash="dash"),
        annotation_text=f"moyenne équipe · {productivite_moyenne:.1f} dossiers/h",
        annotation_position="bottom right", annotation_font=dict(color="#948f80", size=10),
    )

    for nom in ORDRE_SERVICES:
        if service != TOUS_SERVICES and nom != service:
            continue
        d = fusion[fusion["Service_Agent"] == nom]
        if not len(d):
            continue
        fig.add_trace(go.Scatter(
            x=d["anciennete_mois"], y=d["productivite_moy"], mode="markers", name=nom,
            marker=dict(color=SERVICES_COULEURS[nom], size=d["taux_pct"],
                        sizemode="area", sizeref=sizeref, sizemin=DIAMETRE_MIN,
                        line=dict(width=1, color="white")),
            customdata=d[["Matricule_Agent", "taux_pct"]],
            hovertemplate="Agent %{customdata[0]}<br>Ancienneté : %{x:.0f} mois<br>"
                          "Productivité : %{y:.1f} dossiers/h<br>Taux d'anomalie : %{customdata[1]:.1f}%<extra>" + nom + "</extra>",
        ))

    fig.update_xaxes(title_text="Ancienneté (mois)")
    fig.update_yaxes(title_text="Productivité (dossiers/h)")
    return figure_layout_defaults(fig, hauteur=300)


def construire_fig_top_agents(df_anomalies, n=10):
    # Trie par TAUX (proportion de taches atypiques), pas par compte brut :
    # sinon ce classement ne refleterait que qui a traite le plus de taches.
    services_par_agent = df_anomalies["par_agent"].drop_duplicates("Matricule_Agent").set_index("Matricule_Agent")["Service"]
    par_agent = _taux_par_agent(df_anomalies).sort_values("taux_pct", ascending=False).head(n)
    par_agent["Service"] = par_agent["Matricule_Agent"].map(services_par_agent)

    fig = go.Figure(go.Bar(
        y=par_agent["Matricule_Agent"].astype(str) + " · " + par_agent["Service"],
        x=par_agent["taux_pct"], orientation="h", marker_color=SG_ROUGE,
        customdata=par_agent[["nb_anomalies", "nb_taches"]],
        hovertemplate="%{y}<br>%{x:.1f}%% de taches atypiques<br>"
                      "(%{customdata[0]:.0f} sur %{customdata[1]:.0f} taches)<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.update_xaxes(title_text="Taux de tâches atypiques (%)")
    return figure_layout_defaults(fig, hauteur=340)


def layout(df_feat, df_anomalies):
    contenu = [
        html.Div(id="derive-kpi-row", children=construire_kpis(df_feat, df_anomalies)),
        html.Div([
            html.Div([
                html.H3("Productivité vs ancienneté", className="chart-title"),
                html.Div([
                    html.Span("Service :", className="filtre-label"),
                    dcc.Dropdown(
                        id="derive-service-filter",
                        options=[{"label": TOUS_SERVICES, "value": TOUS_SERVICES}] +
                                [{"label": s, "value": s} for s in ORDRE_SERVICES],
                        value=TOUS_SERVICES, clearable=False, style={"width": "220px"},
                    ),
                ], className="filtre-row"),
                html.Div(id="derive-chart-scatter", children=dcc.Graph(
                    figure=construire_fig_scatter(df_feat, df_anomalies), config={"displayModeBar": False})),
                html.P(
                    "Un point sous la ligne pointillée = agent moins productif que la moyenne équipe ; "
                    "plus le point est gros, plus la proportion de tâches atypiques de l'agent est élevée. " + NOTE_ETHIQUE,
                    className="chart-caption",
                ),
            ], className="chart-block"),
            chart_block(
                "Agents les plus atypiques",
                dcc.Graph(figure=construire_fig_top_agents(df_anomalies), config={"displayModeBar": False}),
                NOTE_ETHIQUE,
            ),
        ], className="grid-2"),
        chart_block(
            "Importance des caractéristiques (SHAP)",
            html.Img(src="/model-assets/shap_anomalies_summary.png", style={"maxWidth": "100%", "borderRadius": "3px"}),
            "Ce que le modèle regarde pour juger une tâche atypique ou non.",
        ),
        html.Div([
            html.H3("Détail par agent", className="chart-title"),
            html.Div([
                html.Span("Agent :", className="filtre-label"),
                dcc.Dropdown(
                    id="derive-agent-select",
                    options=[{"label": f"{r.Matricule_Agent} — {r.Service_Agent}", "value": r.Matricule_Agent}
                             for r in df_feat.sort_values("Matricule_Agent").itertuples()],
                    placeholder="Choisir un matricule…", style={"width": "320px"},
                ),
            ], className="filtre-row"),
            html.Div(id="derive-agent-detail"),
        ], className="chart-block"),
    ]
    return page_wrapper(*contenu)
