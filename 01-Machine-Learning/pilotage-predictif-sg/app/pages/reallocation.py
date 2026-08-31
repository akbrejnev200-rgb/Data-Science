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

Le graphique "Service le plus chargé, sur la période" a son propre filtre
independant (fenetre historique uniquement), car lui seul a besoin de
regarder en arriere : il compte, sur la fenetre choisie, combien de jours
chaque service a ete le plus charge. Un desequilibre ponctuel (le service
en tete change souvent) ne justifie pas de reorganiser durablement l'equipe
-- contrairement a un desequilibre structurel (toujours le meme service en
tete) -- une question que ni le KPI (jour cible unique) ni la table de
suggestion (photo instantanee) ne permettent de trancher seuls.
"""

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html

from components.layout_common import chart_block, figure_layout_defaults, kpi_card, kpi_row, page_wrapper
from components.theme import ORDRE_SERVICES, SERVICES_COULEURS
from data.loader import NOM_JOURS, PROCHAINS_JOURS_OUVRES, TODAY, etp_disponible_projete

FENETRES = {"30j": 30, "90j": 90, "tout": None}


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


def _charge_globale_ponderee(moyennes, etp):
    """Charge de reference ponderee par l'ETP de chaque service (pas une
    moyenne simple des 4 charges) : necessaire pour que la reallocation
    suggeree soit a somme nulle -- ce qui est retire aux services
    sur-dotes egale exactement ce qui est ajoute aux services sous-dotes.
    Une moyenne simple (non ponderee) ne garantit pas cette egalite (ecart
    constate en pratique : ~0.1 ETP entre les deux sommes)."""
    total_etp = etp.sum()
    if not total_etp:
        return moyennes.mean()
    return (moyennes * etp).sum() / total_etp


def construire_kpis(djs, fenetre, df_previsions=None, df_agents=None, df_absences=None, taux_repli=None):
    moyennes, etp_moyen, label_etp = _donnees_projection(djs, df_previsions, df_agents, df_absences, taux_repli, fenetre)
    if not len(moyennes):
        return kpi_row(kpi_card("Projection indisponible", "—", "aucune prévision chargée"))

    plus_charge = moyennes.idxmax()
    moins_charge = moyennes.idxmin()
    ecart_pct = (moyennes.max() / moyennes.min() - 1) * 100 if moyennes.min() > 0 else float("nan")
    charge_globale_moy = _charge_globale_ponderee(moyennes, etp_moyen)
    etp_a_deplacer = sum(
        max(0, etp_moyen[s] * (moyennes[s] - charge_globale_moy) / charge_globale_moy)
        for s in moyennes.index
    )

    # "Service le plus/moins charge" retire en tant que KPI dedie : deja
    # visible d'un coup d'oeil sur le graphique juste en dessous (Charge par
    # ETP projetee par service) -- remarque du manager de l'utilisateur,
    # redondance inutile. Les noms des 2 services extremes restent utiles
    # pour comprendre l'ecart : replies dans le sous-texte de l'ecart
    # plutot qu'en KPI a part, et le multiplicateur abstrait (x1.25) devient
    # un pourcentage direct ("+25%") -- se lit comme une phrase, pas comme
    # une formule a interpreter.
    return kpi_row(
        kpi_card("Écart d'équilibre", f"+{ecart_pct:.0f}%", f"{plus_charge} vs {moins_charge}"),
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


def construire_fig_frequence_plus_charge(djs, fenetre_historique):
    """Nombre de jours ou chaque service a ete LE PLUS charge (charge_par_etp
    maximale ce jour-la), sur la fenetre choisie. Repond a la question que
    se pose un manager avant de reorganiser son equipe : un desequilibre
    ponctuel (le service en tete change tout le temps) ne justifie pas de
    deplacer durablement des agents, contrairement a un desequilibre
    structurel (toujours le meme service en tete). Le KPI "Service le plus
    charge" en haut de page ne montre que le jour cible selectionne ; ce
    graphique montre si cette situation est habituelle ou exceptionnelle."""
    fig = go.Figure()
    d = _filtrer_fenetre(djs, fenetre_historique)
    par_jour = d.pivot_table(index="date", columns="Service", values="charge_par_etp")
    par_jour = par_jour.dropna(how="all")
    if not len(par_jour):
        fig.add_annotation(text="Donnée indisponible", showarrow=False)
        return figure_layout_defaults(fig)
    plus_charge_par_jour = par_jour.idxmax(axis=1)
    total_jours = len(plus_charge_par_jour)
    comptes = plus_charge_par_jour.value_counts().reindex(ORDRE_SERVICES).fillna(0).astype(int)

    fig.add_trace(go.Bar(
        x=comptes.index, y=comptes.values,
        marker_color=[SERVICES_COULEURS[s] for s in comptes.index], opacity=0.85,
        text=[f"{v} j" for v in comptes.values], textposition="outside",
        hovertemplate=f"%{{x}}<br>Service le plus chargé %{{y}} jours sur {total_jours}<extra></extra>",
    ))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text=f"Jours en tête (sur {total_jours})")
    return figure_layout_defaults(fig)


def construire_table_reallocation(djs, fenetre, df_previsions=None, df_agents=None, df_absences=None, taux_repli=None):
    moyennes_charge, moyennes_etp, _ = _donnees_projection(djs, df_previsions, df_agents, df_absences, taux_repli, fenetre)
    if not len(moyennes_charge):
        return html.P("Projection indisponible.", className="table-note")

    charge_globale_moy = _charge_globale_ponderee(moyennes_charge, moyennes_etp)
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


CAPTION_FREQUENCE = (
    "Toujours le même service en tête : déséquilibre structurel, qui justifie de déplacer des "
    "agents durablement. Le service en tête change souvent : déséquilibre ponctuel, pas la peine "
    "de réorganiser pour ça."
)

OPTIONS_FILTRE = [
    {"label": f"J+{h} · {NOM_JOURS[d.weekday()][:3]} {d.day:02d}/{d.month:02d}", "value": f"projection_j{h}"}
    for h, d in enumerate(PROCHAINS_JOURS_OUVRES, start=1)
]

VALEUR_FILTRE_DEFAUT = "projection_j1"


OPTIONS_FENETRE_FREQ = [
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

    filtre_freq = html.Div([
        html.Span("Période :", className="filtre-label"),
        dcc.Dropdown(
            id="reallocation-freq-window", options=OPTIONS_FENETRE_FREQ,
            value="90j", clearable=False, style={"width": "200px"},
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
        filtre_freq,
        html.Div(id="reallocation-chart-freq", children=chart_block(
            "Service le plus chargé, sur la période",
            dcc.Graph(figure=construire_fig_frequence_plus_charge(djs, "90j"),
                      config={"displayModeBar": False}),
            CAPTION_FREQUENCE,
        )),
    ]
    return page_wrapper(*contenu)
