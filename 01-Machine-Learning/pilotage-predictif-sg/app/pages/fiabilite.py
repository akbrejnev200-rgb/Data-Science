"""
pages/fiabilite.py

Page dediee aux metriques de performance des 4 modeles, y compris leurs
limites. Les 5 valeurs de comparaison aux baselines naives sont codees en
dur depuis docs/AMELIORATION_MODELES.md (reproduire dynamiquement le split
70/15/15 exact est hors perimetre de la couche app).
"""

import plotly.graph_objects as go
from dash import dcc, html

from components.layout_common import chart_block, figure_layout_defaults, kpi_card, kpi_row, page_wrapper
from components.theme import FAINT, SEQUENTIEL_BLEU, SG_ROUGE

BASELINES_VOLUMES = [
    ("Naïf — moyenne train", 1224.6, False),
    ("Naïf — veille", 770.3, False),
    ("Naïf — même jour sem.", 1050.4, False),
    ("Naïf — moy. mobile 7j", 647.0, True),
    ("Ensemble (retenu)", 647.1, True),
]


CLE_MAE_PAR_MODELE = {
    "Prophet": "mae_test_prophet", "SARIMA": "mae_test_sarima",
    "XGBoost": "mae_test_xgboost", "Ensemble": "mae_test_ensemble",
}


def construire_kpis(m_volumes, m_charge, m_retard, m_anomaly, taux_anomalies_global):
    # Affiche le MAE du modele REELLEMENT retenu (m_volumes["modele_retenu"]),
    # pas toujours "Ensemble" en dur -- ce n'etait plus vrai depuis que
    # XGBoost seul bat l'Ensemble sur le dataset 2022-2026.
    modele_retenu = m_volumes["modele_retenu"]
    mae_ens = m_volumes[CLE_MAE_PAR_MODELE.get(modele_retenu, "mae_test_ensemble")]
    return kpi_row(
        kpi_card("Prévision des volumes", f"{mae_ens:.0f}", f"dossiers/j d'erreur ({modele_retenu})", statut="avertissement"),
        kpi_card("Charge par ETP", f"{m_charge['mae_test']:.1f}", f"dossiers/ETP ({m_charge['type']})"),
        kpi_card("Risque de retard", f"{m_retard['accuracy_test']*100:.1f}%", "accuracy"),
        kpi_card("Détection d'anomalies", f"{m_anomaly['taux_detection_test']*100:.0f}%",
                  f"{taux_anomalies_global*100:.1f}% du volume atypique"),
    )


def construire_fig_baselines():
    labels = [b[0] for b in BASELINES_VOLUMES]
    valeurs = [b[1] for b in BASELINES_VOLUMES]
    couleurs = [SG_ROUGE if b[2] else FAINT for b in BASELINES_VOLUMES]
    fig = go.Figure(go.Bar(
        x=labels, y=valeurs, marker_color=couleurs,
        text=[f"{v:.0f}" for v in valeurs], textposition="outside",
        hovertemplate="%{x}<br>MAE : %{y:.1f}<extra></extra>",
    ))
    fig.update_yaxes(title_text="MAE (dossiers/j)")
    fig.update_layout(showlegend=False)
    return figure_layout_defaults(fig, hauteur=300)


def construire_fig_charge_backtest(df_charge_pred, split="tous"):
    d = df_charge_pred if split == "tous" else df_charge_pred[df_charge_pred["split"] == split]
    couleurs_split = {"train": "#c3c2b7", "validation": "#a3660a", "test": SG_ROUGE}
    fig = go.Figure()
    for s in ["train", "validation", "test"]:
        ds = d[d["split"] == s]
        if not len(ds):
            continue
        fig.add_trace(go.Scatter(
            x=ds["charge_reelle"], y=ds["charge_predite"], mode="markers", name=s,
            marker=dict(color=couleurs_split[s], size=6, opacity=0.6),
            hovertemplate="Réel : %{x:.0f}<br>Prédit : %{y:.0f}<extra>" + s + "</extra>",
        ))
    bornes = [d["charge_reelle"].min(), d["charge_reelle"].max()]
    fig.add_trace(go.Scatter(x=bornes, y=bornes, mode="lines", line=dict(color="#14130f", width=1, dash="dot"),
                              name="Idéal", hoverinfo="skip"))
    fig.update_xaxes(title_text="Charge réelle")
    fig.update_yaxes(title_text="Charge prédite")
    return figure_layout_defaults(fig)


def construire_fig_confusion(df_retard_conf, split="test"):
    d = df_retard_conf if split == "tous" else df_retard_conf[df_retard_conf["split"] == split]
    matrice = d.groupby(["est_en_retard_reel", "est_en_retard_predit"])["n"].sum().unstack()
    matrice = matrice.reindex(index=[0, 1], columns=[0, 1]).fillna(0)
    fig = go.Figure(go.Heatmap(
        z=matrice.values, x=["Prédit : à temps", "Prédit : en retard"], y=["Réel : à temps", "Réel : en retard"],
        colorscale=[[0, "#ffffff"], [1, SEQUENTIEL_BLEU[4]]],
        text=matrice.values, texttemplate="%{text:,.0f}", textfont=dict(size=13),
        hovertemplate="%{y} / %{x}<br>%{z:,.0f} tâches<extra></extra>", showscale=False,
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return figure_layout_defaults(fig, hauteur=300)


def layout(dj, df_previsions, df_charge_pred, df_retard_conf, m_volumes, m_charge, m_retard, m_anomaly,
           taux_anomalies_global):
    contenu = [
        html.Div(id="fiabilite-kpi-row", children=construire_kpis(
            m_volumes, m_charge, m_retard, m_anomaly, taux_anomalies_global,
        )),
        chart_block(
            "Prévision des volumes vs référence naïve",
            dcc.Graph(figure=construire_fig_baselines(), config={"displayModeBar": False}),
            "Le modèle retenu ne bat une simple moyenne mobile 7j que de 0,3% — l'essentiel de la variation "
            "quotidienne relève de facteurs métier non présents dans les données.",
        ),
        html.Div([
            html.H3("Charge par ETP : réel vs prédit", className="chart-title"),
            html.Div([
                dcc.RadioItems(
                    id="fiabilite-split-filter",
                    options=[{"label": " Toutes", "value": "tous"}, {"label": " Entraînement", "value": "train"},
                             {"label": " Validation", "value": "validation"}, {"label": " Test", "value": "test"}],
                    value="test", inline=True, style={"display": "flex", "gap": "16px"},
                ),
            ], className="filtre-row"),
            html.Div(id="fiabilite-chart-charge-graph", children=dcc.Graph(
                figure=construire_fig_charge_backtest(df_charge_pred, "test"), config={"displayModeBar": False})),
            html.P(
                "Chaque point est un jour ; plus il est proche de la diagonale \"Idéal\", plus la prévision était "
                "exacte. Par défaut sur Test (données jamais vues à l'entraînement, cohérent avec le MAE affiché "
                "ci-dessus) — Train serait plus optimiste, le modèle les ayant déjà vues.",
                className="chart-caption",
            ),
        ], className="chart-block"),
        html.Div([
            html.H3("Risque de retard : matrice de confusion", className="chart-title"),
            html.Div([
                dcc.RadioItems(
                    id="fiabilite-retard-split-filter",
                    options=[{"label": " Toutes", "value": "tous"}, {"label": " Entraînement", "value": "train"},
                             {"label": " Validation", "value": "validation"}, {"label": " Test", "value": "test"}],
                    value="test", inline=True, style={"display": "flex", "gap": "16px"},
                ),
            ], className="filtre-row"),
            html.Div(id="fiabilite-chart-retard-graph", children=dcc.Graph(
                figure=construire_fig_confusion(df_retard_conf), config={"displayModeBar": False})),
            html.P(
                "Par défaut sur Test (données jamais vues à l'entraînement, cohérent avec l'accuracy affichée "
                "ci-dessus) — Train inclut des tâches déjà vues par le modèle, donc plus optimiste.",
                className="chart-caption",
            ),
        ], className="chart-block"),
    ]
    return page_wrapper(*contenu)
