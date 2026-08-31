"""callbacks/cb_fiabilite.py -- filtrage par split (train/validation/test)
des graphiques charge (reel vs predit) et retard (matrice de confusion),
pandas pur sur df_charge_pred/df_retard_conf deja charges."""

from dash import Input, Output, dcc

from pages.fiabilite import construire_fig_charge_backtest, construire_fig_confusion


def register(app, dj, df_previsions, df_charge_pred, df_retard_conf, m_volumes, m_charge, m_retard, m_anomaly):
    @app.callback(
        Output("fiabilite-chart-charge-graph", "children"),
        Input("fiabilite-split-filter", "value"),
    )
    def maj_charge(split):
        return dcc.Graph(figure=construire_fig_charge_backtest(df_charge_pred, split), config={"displayModeBar": False})

    @app.callback(
        Output("fiabilite-chart-retard-graph", "children"),
        Input("fiabilite-retard-split-filter", "value"),
    )
    def maj_retard(split):
        return dcc.Graph(figure=construire_fig_confusion(df_retard_conf, split), config={"displayModeBar": False})
