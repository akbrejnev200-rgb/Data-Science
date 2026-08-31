"""
pages/reallocation.py

Compare la charge par ETP entre services et propose un rééquilibrage
illustratif — calcul simple, jamais une sortie de modèle.

Le filtre "Période" (KPI + boxplot + table) ne propose que des jours de
prévision, J+1 à J+30 -- pas d'historique passé, seule la prévision
compte pour cette page. Pour chaque jour, la charge par service est
calculee avec le meme ETP reel (contrats + conges, cf. page Surcharge) que
pour ce jour cible exact -- pas une moyenne recente. Seule la repartition
du volume par service reste illustrative (part historique) : aucun modele
n'a ete entraine par service (teste et ecarte, cf.
docs/AMELIORATION_MODELES.md -- un SARIMA par service s'est revele 22.8%
pire qu'une simple moyenne mobile).

Le graphique "Évolution de l'écart entre services" a son propre filtre
independant (fenetre historique + sous-ensemble de services), car lui
seul a besoin de regarder en arriere pour juger si le desequilibre
s'aggrave ou se resorbe.
"""

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html

from components.layout_common import chart_block, figure_layout_defaults, kpi_card, kpi_row, page_wrapper
from components.theme import ORDRE_SERVICES, SERVICES_COULEURS, SG_ROUGE
from data.loader import NOM_JOURS, PROCHAINS_JOURS_OUVRES, TODAY, etp_disponible_projete

FENETRES = {"30j": 30, "90j": 90, "tout": None}
FENETRE_ETP_PROJECTION = "90j"  # fenetre stable pour l'ETP moyen recent, utilisee par le graphique d'ecart uniquement


def _jour_projection(fenetre):
    """Retourne le numero de jour (1-30) encode dans fenetre (ex
    "projection_j5" -> 5), ou None si invalide."""
    if fenetre and fenetre.startswith("projection_j"):
        try:
            return int(fenetre[len("projection_j"):])
        except ValueError:
            return None
    return None


def _filtrer_fenetre(djs, fenetre):
    if FENETRES.get(fenetre) is None:
        return djs
    date_min = pd.Timestamp(TODAY) - timedelta(days=FENETRES[fenetre])
    return djs[djs["date"] >= date_min]


def _charge_projetee_jour(djs, df_previsions, df_agents, df_absences, taux_repli, jour):
    """Charge par ETP projetee pour un jour precis (pas une moyenne) :
    volume prevu de ce jour reparti par part historique de service, divise
    par l'ETP reel de ce service a cette date (contrats + conges reels si
    couverts, sinon taux de repli historique par type de contrat) -- meme
    logique que la page Surcharge, plutot qu'un ETP moyen sur 90 jours."""
    vide = (pd.Series(dtype=float), pd.Series(dtype=float), None, False)
    if df_previsions is None or not len(df_previsions) or df_agents is None:
        return vide
    if jour < 1 or jour > len(PROCHAINS_JOURS_OUVRES):
        return vide
    ligne = df_previsions[df_previsions["horizon_j"] == jour]
    if not len(ligne):
        return vide
    date_cible = PROCHAINS_JOURS_OUVRES[jour - 1]
    volume_jour = ligne["volume_prevu"].iloc[0]

    part_volume = djs.groupby("Service")["volume_entrant_jour"].sum()
    part_volume = part_volume / part_volume.sum()
    etp_global, etp_service, couverture_reelle = etp_disponible_projete(
        date_cible, df_agents, df_absences, taux_repli,
    )

    charges = {}
    for s in ORDRE_SERVICES:
        if s not in etp_service.index or etp_service[s] <= 0:
            continue
        charges[s] = (volume_jour * part_volume.get(s, 0)) / etp_service[s]
    return pd.Series(charges), etp_service.reindex(ORDRE_SERVICES), date_cible, couverture_reelle


def _donnees_projection(djs, df_previsions, df_agents, df_absences, taux_repli, fenetre):
    """Point d'entree unique pour KPI / boxplot / table : charge par
    service pour le jour precis demande, et libelle expliquant la source
    de l'ETP utilise (reel si couvert par les conges, sinon estime)."""
    jour = _jour_projection(fenetre)
    if not jour:
        return pd.Series(dtype=float), pd.Series(dtype=float), None
    charges, etp_ref, date_cible, couverture_reelle = _charge_projetee_jour(
        djs, df_previsions, df_agents, df_absences, taux_repli, jour,
    )
    label_etp = "réel (contrats + congés)" if couverture_reelle else "estimé (taux d'absence historique)"
    return charges, etp_ref, label_etp


def construire_kpis(djs, fenetre, df_previsions=None, df_agents=None, df_absences=None, taux_repli=None):
    moyennes, etp_moyen, label_etp = _donnees_projection(djs, df_previsions, df_agents, df_absences, taux_repli, fenetre)
    if not len(moyennes):
        return kpi_row(kpi_card("Projection indisponible", "—", "aucune prévision chargée"))

    plus_charge = moyennes.idxmax()
    moins_charge = moyennes.idxmin()
    ecart = moyennes.max() / moyennes.min() if moyennes.min() > 0 else float("nan")
    charge_globale_moy = moyennes.mean()
    etp_a_deplacer = sum(
        max(0, etp_moyen[s] * (moyennes[s] - charge_globale_moy) / charge_globale_moy)
        for s in moyennes.index
    )

    return kpi_row(
        kpi_card("Service le plus chargé", plus_charge, f"{moyennes[plus_charge]:.0f} dossiers/ETP", statut="alerte"),
        kpi_card("Service le moins chargé", moins_charge, f"{moyennes[moins_charge]:.0f} dossiers/ETP", statut="bon"),
        kpi_card("Écart d'équilibre", f"×{ecart:.2f}", "charge max / charge min"),
        kpi_card("ETP à redéployer (est.)", f"≈ {etp_a_deplacer:.1f}", f"ETP {label_etp}" if label_etp else "pour équilibrer"),
    )


def construire_fig_charge_boxplot(djs, fenetre, df_previsions=None, df_agents=None, df_absences=None, taux_repli=None):
    fig = go.Figure()
    moyennes, _, _ = _donnees_projection(djs, df_previsions, df_agents, df_absences, taux_repli, fenetre)
    if not len(moyennes):
        fig.add_annotation(text="Projection indisponible", showarrow=False)
        return figure_layout_defaults(fig)
    fig.add_trace(go.Bar(
        x=moyennes.index, y=moyennes.values, name="Projection",
        marker_color=[SERVICES_COULEURS[s] for s in moyennes.index], opacity=0.85,
        hovertemplate="%{x}<br>Projeté : %{y:.0f} dossiers/ETP<extra></extra>",
    ))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="Dossiers / ETP (projeté)")
    return figure_layout_defaults(fig)


def _projeter_ecart_futur(djs, d, df_previsions, horizon_j, services):
    prev = df_previsions[df_previsions["horizon_j"] <= horizon_j]
    if not len(prev):
        return pd.Series(dtype=float)
    part_volume = djs[djs["Service"].isin(services)].groupby("Service")["volume_entrant_jour"].sum()
    part_volume = part_volume / part_volume.sum()
    etp_moyen = d[d["Service"].isin(services)].groupby("Service")["etp_disponible"].mean()

    projection = pd.DataFrame(index=prev["date"].values)
    for s in services:
        if s not in etp_moyen.index or etp_moyen[s] <= 0:
            continue
        volume_service = prev.set_index("date")["volume_prevu"].values * part_volume.get(s, 0)
        projection[s] = volume_service / etp_moyen[s]
    if projection.shape[1] < 2:
        return pd.Series(dtype=float)
    return projection.max(axis=1) / projection.min(axis=1)


HORIZON_PROJECTION_ECART = 30  # portion pointillee toujours affichee jusqu'a J+30


def construire_fig_ecart_temporel(djs, fenetre_historique, df_previsions=None, services=None):
    # Ni le graphique de charge (moyenne+dispersion sur la periode) ni le
    # tableau (photo instantanee) ne montrent si le desequilibre s'aggrave
    # ou se resorbe -- seule une serie temporelle de l'ecart repond a ca.
    # Filtre independant du reste de la page : fenetre historique (30/90j/
    # tout) + sous-ensemble de services (l'ecart max/min ne se calcule que
    # sur les services coches, au moins 2 necessaires).
    services = services or ORDRE_SERVICES
    fig = go.Figure()

    if len(services) < 2:
        fig.add_annotation(text="Sélectionnez au moins 2 services", showarrow=False)
        return figure_layout_defaults(fig)

    d = _filtrer_fenetre(djs[djs["Service"].isin(services)], fenetre_historique)
    par_jour = d.pivot_table(index="date", columns="Service", values="charge_par_etp")
    ecart_brut = par_jour.max(axis=1) / par_jour.min(axis=1)
    ecart_lisse = ecart_brut.rolling(window=7, min_periods=1).mean()

    fig.add_trace(go.Scatter(
        x=ecart_brut.index, y=ecart_brut.values, mode="lines", name="Écart quotidien",
        line=dict(color="#e5e1d5", width=1),
        hovertemplate="%{x|%d/%m/%Y}<br>×%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ecart_lisse.index, y=ecart_lisse.values, mode="lines", name="Moyenne mobile 7j",
        line=dict(color=SG_ROUGE, width=2),
        hovertemplate="%{x|%d/%m/%Y}<br>×%{y:.2f} (moy. 7j)<extra></extra>",
    ))

    if df_previsions is not None and len(df_previsions):
        d_etp = _filtrer_fenetre(djs, FENETRE_ETP_PROJECTION)
        ecart_projete = _projeter_ecart_futur(djs, d_etp, df_previsions, HORIZON_PROJECTION_ECART, services)
        if len(ecart_projete):
            fig.add_trace(go.Scatter(
                x=ecart_projete.index, y=ecart_projete.values, mode="lines", name="Projection (illustrative)",
                line=dict(color=SG_ROUGE, width=1.6, dash="dot"),
                hovertemplate="%{x|%d/%m/%Y}<br>×%{y:.2f} (projeté)<extra></extra>",
            ))

    fig.add_hline(y=1, line=dict(color="#c3c2b7", width=1, dash="dash"),
                   annotation_text="équilibre parfait", annotation_position="bottom right",
                   annotation_font=dict(color="#948f80", size=10))
    fig.update_yaxes(title_text="Écart (charge max ÷ charge min)")
    return figure_layout_defaults(fig)


def construire_table_reallocation(djs, fenetre, df_previsions=None, df_agents=None, df_absences=None, taux_repli=None):
    moyennes_charge, moyennes_etp, _ = _donnees_projection(djs, df_previsions, df_agents, df_absences, taux_repli, fenetre)
    if not len(moyennes_charge):
        return html.P("Projection indisponible.", className="table-note")

    charge_globale_moy = moyennes_charge.mean()
    lignes = []
    for s in moyennes_charge.index:
        etp_actuel = moyennes_etp[s]
        etp_suggere = etp_actuel * (moyennes_charge[s] / charge_globale_moy) if charge_globale_moy else etp_actuel
        lignes.append({
            "Service": s,
            "Charge (dossiers/ETP)": round(moyennes_charge[s], 1),
            "ETP actuel": round(etp_actuel, 1),
            "ETP suggéré": round(etp_suggere, 1),
            "Delta": round(etp_suggere - etp_actuel, 1),
        })

    return dash_table.DataTable(
        data=lignes,
        columns=[{"name": c, "id": c} for c in lignes[0].keys()],
        style_cell={"fontFamily": "-apple-system, sans-serif", "fontSize": "12.5px", "padding": "9px 12px",
                     "border": "none", "fontVariantNumeric": "tabular-nums"},
        style_cell_conditional=[
            {"if": {"column_id": c}, "textAlign": "right"}
            for c in ["Charge (dossiers/ETP)", "ETP actuel", "ETP suggéré", "Delta"]
        ],
        style_header={"fontWeight": "600", "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": "0.03em",
                       "backgroundColor": "#f6f4ee", "color": "#6b6759", "border": "none",
                       "borderBottom": "1px solid #e5e1d5"},
        style_data={"border": "none", "borderBottom": "1px solid #efece2", "color": "#3a372f"},
        style_as_list_view=True,
    )


CAPTION_BOXPLOT_JOUR = (
    "Valeur projetée pour ce jour précis (volume prévu ÷ ETP réel/estimé de ce jour)."
)


def caption_boxplot(fenetre):
    return CAPTION_BOXPLOT_JOUR


def titre_boxplot(fenetre):
    jour = _jour_projection(fenetre)
    return f"Charge par ETP projetée par service (J+{jour})" if jour else "Charge par ETP projetée par service"


CAPTION_ECART = (
    "Écart qui se creuse : déséquilibre structurel. Écart qui se resorbe : pic ponctuel. "
    "Portion pointillée : projection illustrative, pas une sortie de modèle."
)

OPTIONS_FILTRE = [
    {"label": f"J+{h} · {NOM_JOURS[d.weekday()][:3]} {d.day:02d}/{d.month:02d}", "value": f"projection_j{h}"}
    for h, d in enumerate(PROCHAINS_JOURS_OUVRES, start=1)
]

VALEUR_FILTRE_DEFAUT = "projection_j1"


OPTIONS_FENETRE_ECART = [
    {"label": "30 derniers jours", "value": "30j"},
    {"label": "90 derniers jours", "value": "90j"},
    {"label": "Historique complet", "value": "tout"},
]


def layout(djs, df_previsions=None, df_agents=None, df_absences=None, taux_repli=None):
    filtre_jour = html.Div([
        html.Span("Prévision pour :", className="filtre-label"),
        dcc.Dropdown(
            id="reallocation-window", options=OPTIONS_FILTRE,
            value=VALEUR_FILTRE_DEFAUT, clearable=False, style={"width": "280px"},
        ),
    ], className="filtre-row")

    filtre_ecart = html.Div([
        html.Span("Période :", className="filtre-label"),
        dcc.Dropdown(
            id="reallocation-ecart-window", options=OPTIONS_FENETRE_ECART,
            value="90j", clearable=False, style={"width": "200px"},
        ),
        html.Span("Services :", className="filtre-label"),
        dcc.Dropdown(
            id="reallocation-ecart-services", options=[{"label": s, "value": s} for s in ORDRE_SERVICES],
            value=ORDRE_SERVICES, multi=True, clearable=False, style={"width": "420px"},
        ),
    ], className="filtre-row")

    args = (djs, VALEUR_FILTRE_DEFAUT, df_previsions, df_agents, df_absences, taux_repli)

    contenu = [
        filtre_jour,
        html.Div(id="reallocation-kpi-row", children=construire_kpis(*args)),
        html.Div(id="reallocation-chart-box", children=chart_block(
            titre_boxplot(VALEUR_FILTRE_DEFAUT),
            dcc.Graph(figure=construire_fig_charge_boxplot(*args), config={"displayModeBar": False}),
            caption_boxplot(VALEUR_FILTRE_DEFAUT),
        )),
        html.Div(id="reallocation-table-block", children=chart_block(
            "Suggestion de rééquilibrage",
            construire_table_reallocation(*args),
            "Calcul illustratif d'aide à la décision — pas une sortie du modèle.",
        )),
        filtre_ecart,
        html.Div(id="reallocation-chart-ecart", children=chart_block(
            "Évolution de l'écart entre services",
            dcc.Graph(figure=construire_fig_ecart_temporel(djs, "90j", df_previsions, ORDRE_SERVICES),
                      config={"displayModeBar": False}),
            CAPTION_ECART,
        )),
    ]
    return page_wrapper(*contenu)
