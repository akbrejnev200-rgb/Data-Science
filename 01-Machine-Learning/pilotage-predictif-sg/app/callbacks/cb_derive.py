"""callbacks/cb_derive.py -- filtrage par service du nuage de points
(pandas pur), et requete SQL live bornee a un agent au moment de sa
selection dans le menu deroulant de detail."""

import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from components.layout_common import figure_layout_defaults, kpi_row, kpi_card
from components.theme import SG_ROUGE
from data.loader import charger_historique_agent
from pages.derive import construire_fig_scatter


def register(app, df_feat, df_anomalies):
    @app.callback(
        Output("derive-chart-scatter", "children"),
        Input("derive-service-filter", "value"),
    )
    def maj_scatter(service):
        return dcc.Graph(
            figure=construire_fig_scatter(df_feat, df_anomalies, service), config={"displayModeBar": False},
        )

    @app.callback(
        Output("derive-agent-detail", "children"),
        Input("derive-agent-select", "value"),
    )
    def maj_detail_agent(matricule):
        if not matricule:
            return html.P("Aucun agent sélectionné.", className="table-note")

        info = df_feat[df_feat["Matricule_Agent"] == matricule].iloc[0]
        anomalies_agent = df_anomalies["par_agent"]
        anomalies_agent = anomalies_agent[anomalies_agent["Matricule_Agent"] == matricule]
        nb_anomalies = int(anomalies_agent["nb_anomalies"].sum()) if len(anomalies_agent) else 0

        historique = charger_historique_agent(matricule)
        historique = historique.dropna(subset=["productivite_dossiers_par_heure"]).sort_values("Date_Creation")

        fig = go.Figure(go.Scatter(
            x=historique["Date_Creation"], y=historique["productivite_dossiers_par_heure"],
            mode="lines+markers", line=dict(color=SG_ROUGE, width=1.5), marker=dict(size=4),
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f} dossiers/h<extra></extra>",
        ))
        fig.update_yaxes(title_text="Productivité (dossiers/heure)")
        fig = figure_layout_defaults(fig, hauteur=280)

        kpis = kpi_row(
            kpi_card("Productivité moyenne (agent)", f"{info['productivite_moy']:.1f}", "dossiers / heure"),
            kpi_card("Productivité moyenne (équipe)", f"{df_feat['productivite_moy'].mean():.1f}", "toutes équipes"),
            kpi_card("Tâches atypiques", nb_anomalies, statut="avertissement" if nb_anomalies else "bon"),
        )
        return html.Div([kpis, dcc.Graph(figure=fig, config={"displayModeBar": False})])
