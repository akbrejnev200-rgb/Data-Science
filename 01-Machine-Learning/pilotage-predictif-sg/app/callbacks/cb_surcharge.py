"""callbacks/cb_surcharge.py -- filtrage par service (tableau de
previsions), par horizon (graphique de prevision) et par jour cible
(KPI + charge par service + alertes prevues, J+1 a J+30)."""

import pandas as pd
from dash import Input, Output, dcc

from components.layout_common import chart_block
from pages.surcharge import (
    FENETRE_TACHES_EN_COURS_JOURS, bloc_grid_prevision, caption_previsions_service,
    construire_donnees_prevision, construire_fig_forecast, construire_kpis,
    construire_risque_surcharge_fenetre, construire_table_previsions_service,
    construire_table_risque_retard,
)


def register(app, dj, djs, df_previsions, m_charge, df_agents, df_absences, taux_repli, df_risque_retard=None):
    @app.callback(
        Output("surcharge-chart-forecast", "children"),
        Input("surcharge-horizon-filter", "value"),
    )
    def maj_forecast(horizon_j):
        return chart_block(
            "", dcc.Graph(figure=construire_fig_forecast(dj, df_previsions, horizon_j), config={"displayModeBar": False}),
            "Modèle Ensemble (SARIMA + XGBoost), erreur moyenne ≈ 647 dossiers/jour — détail sur la page Fiabilité.",
        )

    @app.callback(
        Output("surcharge-table-previsions", "children"),
        Input("surcharge-table-service-filter", "value"),
    )
    def maj_table_previsions(service):
        titre = "Volume prévu, jour par jour (J+1 à J+7)" if service == "Tous services" \
            else f"Volume prévu — {service}, jour par jour (J+1 à J+7)"
        return chart_block(
            titre, construire_table_previsions_service(djs, df_previsions, service),
            caption_previsions_service(service),
        )

    @app.callback(
        Output("surcharge-kpi-row", "children"),
        Output("surcharge-prevision-grid", "children"),
        Input("surcharge-jour-prevision", "value"),
    )
    def maj_jour_prevision(horizon_j):
        prevision = construire_donnees_prevision(
            dj, djs, df_previsions, m_charge, df_agents, df_absences, taux_repli, horizon_j,
        )
        risque_fenetre = construire_risque_surcharge_fenetre(
            dj, djs, df_previsions, m_charge, df_agents, df_absences, taux_repli, horizon_j,
        )
        return construire_kpis(prevision, risque_fenetre, df_risque_retard), bloc_grid_prevision(djs, prevision)

    @app.callback(
        Output("surcharge-table-retard", "children"),
        Input("surcharge-retard-service-filter", "value"),
    )
    def maj_table_retard(service):
        return chart_block(
            "", construire_table_risque_retard(
                df_risque_retard if df_risque_retard is not None else pd.DataFrame(), service,
            ),
            f"Tâches actuellement en cours, créées dans les {FENETRE_TACHES_EN_COURS_JOURS} derniers jours, "
            "scorées en direct par le modèle de risque de retard (Bloc 3C) — triée par risque décroissant.",
        )
