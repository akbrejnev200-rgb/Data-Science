"""
pages/derive.py

Detection d'anomalies au niveau tache/agent — signal pour un manager,
jamais une decision RH automatique.
"""

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from components.layout_common import chart_block, figure_layout_defaults, kpi_card, kpi_row, page_wrapper
from components.theme import ORDRE_SERVICES, SERVICES_COULEURS, SG_ROUGE
from data.loader import MODELS_DIR

NOTE_ETHIQUE = "Signal à vérifier avec l'agent, jamais une évaluation automatisée."

# Libelles lisibles pour les 7 variables du modele d'anomalies (Bloc 3D) --
# les noms techniques (snake_case, franglais) parlent a un data scientist,
# pas a un manager en soutenance.
LIBELLES_FEATURES_SHAP = {
    "coeff_productivite_contrat": "Coefficient de productivité du contrat",
    "delai_prise_en_charge_j": "Délai de prise en charge (jours)",
    "Volume_Dossiers": "Volume de dossiers traités",
    "complexite_num": "Complexité du dossier (score 1-3)",
    "duree_traitement_reelle_j": "Durée de traitement réelle (jours)",
    "productivite_dossiers_par_heure": "Productivité (dossiers / heure)",
    "Temps_Passe_Declare_Min": "Temps déclaré (minutes)",
}

def _taux_par_agent(df_anomalies):
    par_agent = df_anomalies["par_agent"].groupby("Matricule_Agent").agg(
        nb_taches=("nb_taches_evaluees", "sum"), nb_anomalies=("nb_anomalies", "sum"),
    ).reset_index()
    par_agent["taux_pct"] = par_agent["nb_anomalies"] / par_agent["nb_taches"] * 100
    return par_agent


def construire_kpis(df_feat, df_anomalies):
    par_agent = _taux_par_agent(df_anomalies)
    n_agents = len(df_feat)

    # "Productivite moyenne" et "Agents a taux anormal" retires (le 2e etait
    # un classement d'agents individuels par taux, juge pas assez ethique --
    # remarque de l'utilisateur). Ne restent que des KPI qui ne pointent pas
    # d'agent precis.
    return kpi_row(
        kpi_card("Agents suivis", n_agents, "au total"),
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


def construire_fig_shap_importance():
    """Remplace le graphique SHAP "beeswarm" (image brute shap.summary_plot,
    pensee pour un public data science : nuage de points colore avec une
    legende "feature value low/high" a interpreter) par un classement en
    barres des variables par importance moyenne -- meme logique que "Agents
    les plus atypiques" juste au-dessus, deja valide comme lisible. Les
    valeurs viennent de models/shap_anomalies_importance.csv, exportees par
    modelisation_anomalies.py (moyenne de |valeur SHAP| par variable sur un
    echantillon de 2000 taches, Bloc 3D)."""
    chemin = MODELS_DIR / "shap_anomalies_importance.csv"
    if not chemin.exists():
        return None
    df_imp = pd.read_csv(chemin).sort_values("importance_moyenne_abs", ascending=False)
    df_imp["libelle"] = df_imp["feature"].map(lambda f: LIBELLES_FEATURES_SHAP.get(f, f))

    fig = go.Figure(go.Bar(
        y=df_imp["libelle"], x=df_imp["importance_moyenne_abs"], orientation="h", marker_color=SG_ROUGE,
        hovertemplate="%{y}<br>Importance moyenne : %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.update_xaxes(title_text="Importance moyenne (impact sur la décision du modèle)")
    return figure_layout_defaults(fig, hauteur=300)


def layout(df_feat, df_anomalies):
    fig_shap = construire_fig_shap_importance()
    contenu = [
        html.Div(id="derive-kpi-row", children=construire_kpis(df_feat, df_anomalies)),
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
            "Importance des caractéristiques (SHAP)",
            dcc.Graph(figure=fig_shap, config={"displayModeBar": False}) if fig_shap is not None
            else html.Img(src="/model-assets/shap_anomalies_summary.png", style={"maxWidth": "100%", "borderRadius": "3px"}),
            "Plus la barre est longue, plus cette caractéristique pèse dans la décision du modèle pour repérer "
            "une tâche atypique ou non (Bloc 3D, Isolation Forest).",
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
