"""callbacks/cb_reallocation.py -- deux filtres independants :
"reallocation-window" (jour de prevision J+1..J+30) pilote KPI + boxplot +
table ; "reallocation-ecart-window" + "-services" pilote uniquement le
graphique d'ecart temporel (fenetre historique + sous-ensemble de
services)."""

from dash import Input, Output, dcc

from components.layout_common import chart_block
from pages.reallocation import (
    CAPTION_ECART, caption_boxplot, construire_fig_charge_boxplot,
    construire_fig_ecart_temporel, construire_kpis, construire_table_reallocation, titre_boxplot,
)


def register(app, djs, df_previsions=None, df_agents=None, df_absences=None, taux_repli=None):
    @app.callback(
        Output("reallocation-kpi-row", "children"),
        Output("reallocation-chart-box", "children"),
        Output("reallocation-table-block", "children"),
        Input("reallocation-window", "value"),
    )
    def maj_jour(fenetre):
        args = (djs, fenetre, df_previsions, df_agents, df_absences, taux_repli)

        kpis = construire_kpis(*args)
        bloc_box = chart_block(
            titre_boxplot(fenetre),
            dcc.Graph(figure=construire_fig_charge_boxplot(*args), config={"displayModeBar": False}),
            caption_boxplot(fenetre),
        )
        bloc_table = chart_block(
            "Suggestion de rééquilibrage",
            construire_table_reallocation(*args),
            "Calcul illustratif d'aide à la décision — pas une sortie du modèle.",
        )
        return kpis, bloc_box, bloc_table

    @app.callback(
        Output("reallocation-chart-ecart", "children"),
        Input("reallocation-ecart-window", "value"),
        Input("reallocation-ecart-services", "value"),
    )
    def maj_ecart(fenetre_historique, services):
        return chart_block(
            "Évolution de l'écart entre services",
            dcc.Graph(figure=construire_fig_ecart_temporel(djs, fenetre_historique, df_previsions, services),
                      config={"displayModeBar": False}),
            CAPTION_ECART,
        )
