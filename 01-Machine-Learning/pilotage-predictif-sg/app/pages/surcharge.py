"""
pages/surcharge.py

Page d'accueil, entierement tournee vers l'avenir (J+1 a J+30) : prevision
de volume, charge par service prevue, alertes prevues, taches ouvertes a
risque de retard. Pas de vue retrospective ici -- l'historique/tendance
vit sur la page Reallocation.
"""

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html

from components.layout_common import (
    alert_item, chart_block, figure_layout_defaults, kpi_card, kpi_card_ajuste, kpi_row,
    page_wrapper, section_block, service_bar_row,
)
from components.theme import ORDRE_SERVICES, SERVICES_COULEURS, SG_ROUGE
from data.loader import (
    FENETRE_TACHES_EN_COURS_JOURS, NOM_JOURS, NOM_MOIS, PROCHAINS_JOURS_OUVRES, TODAY,
    construire_serie_prevision,
)

HORIZON_TABLE_JOURS = 7  # tableau "jour par jour" : fixe, choix de lisibilite
HORIZON_JOUR_MAX = len(PROCHAINS_JOURS_OUVRES)  # selecteur "Jour" (KPI) : va jusqu'a la limite du modele, 30

FENETRE_JOURS = 90
TOUS_SERVICES = "Tous services"


def construire_donnees_prevision(dj, djs, df_previsions, m_charge, df_agents, df_absences, taux_repli, horizon_j=1):
    """Calcule tout ce dont les KPI, le bloc de charge par service et les
    alertes ont besoin pour le jour cible (J+horizon_j, 1 a 7). Volume =
    vraie sortie du modele de volume ; charge globale = vraie sortie du
    modele de charge (m_charge.predict(), seule inference live de l'app,
    reprevue recursivement jour apres jour au-dela de J+1) ; ETP = calcul
    reel (contrats + conges) si couvert, sinon estime via un taux
    d'absence historique par type de contrat ; charge par service = seule
    partie illustrative (repartition du volume global par part historique
    de service, cf. page Reallocation)."""
    jours_cibles = PROCHAINS_JOURS_OUVRES[:horizon_j]
    serie = construire_serie_prevision(dj, df_previsions, df_agents, df_absences, taux_repli, m_charge, jours_cibles)
    jour = serie[-1]

    volume_prevu = jour["volume_prevu"]
    volume_prevu_max = jour["volume_prevu_max"]
    etp_service = jour["etp_service"]

    seuils_service = djs.groupby("Service")["charge_par_etp"].quantile(0.75)
    part_volume = djs.groupby("Service")["volume_entrant_jour"].sum()
    part_volume = part_volume / part_volume.sum()

    def _repartir(volume_global):
        charges = {}
        if volume_global is None:
            return pd.Series(charges)
        for s in ORDRE_SERVICES:
            if s not in etp_service.index or etp_service[s] <= 0:
                continue
            charges[s] = (volume_global * part_volume.get(s, 0)) / etp_service[s]
        return pd.Series(charges)

    charges_service = _repartir(volume_prevu)
    # Une prevision est une estimation centrale lissee : comparee au seuil
    # historique (75e percentile de donnees reelles, donc bruitees), elle
    # ne le depasse quasiment jamais, meme les jours a forte charge --
    # constate empiriquement (0/30 jours declenchaient une alerte). Les
    # alertes se basent donc sur la borne haute de l'intervalle de
    # prevision (volume_prevu_max) : "et si le haut de la fourchette se
    # realise ?", plus pertinent pour un signal de risque. Les barres du
    # graphique, elles, restent sur l'estimation centrale (informative).
    charges_service_risque = _repartir(volume_prevu_max)

    return {
        "date_cible": jour["date"],
        "horizon_j": horizon_j,
        "volume_prevu": volume_prevu,
        "etp_global": jour["etp_global"],
        "etp_service": etp_service,
        "couverture_reelle": jour["couverture_reelle"],
        "charge_prevue": jour["charge_prevue"],
        "charges_service": charges_service,
        "charges_service_risque": charges_service_risque,
        "seuils_service": seuils_service,
        "nb_etp_impactes_jour": jour["nb_etp_impactes_jour"],
        "volume_annonce_jour": jour["volume_annonce_jour"],
        "volume_prevu_ajuste": jour["volume_prevu_ajuste"],
        "charge_prevue_ajustee": jour["charge_prevue_ajustee"],
    }


def _carte_charge(prevision):
    """Charge globale/ETP : sortie brute du modele (Bloc 3B), sauf si un
    incident ETP est actif pour ce jour cible -- dans ce cas, affiche aussi
    le chiffre recalcule avec l'ETP disponible reduit d'autant (cf.
    data.loader.construire_serie_prevision), sans jamais remplacer le
    chiffre du modele par le chiffre ajuste."""
    charge_prevue = prevision["charge_prevue"]
    if charge_prevue is None:
        return kpi_card("Charge globale/ETP (prévu)", "—", "dossiers, modèle Bloc 3B")

    nb_etp_impactes = prevision.get("nb_etp_impactes_jour") or 0
    if nb_etp_impactes:
        return kpi_card_ajuste(
            "Charge globale/ETP (prévu)", f"{charge_prevue:.0f}", f"{prevision['charge_prevue_ajustee']:.0f}",
            f"ajusté : {nb_etp_impactes:g} ETP en moins signalé(s)",
        )
    return kpi_card("Charge globale/ETP (prévu)", f"{charge_prevue:.0f}", "dossiers, modèle Bloc 3B")


def _carte_volume(prevision):
    """Volume entrant : sortie brute du modele (Ensemble), sauf si une
    annonce est active pour ce jour cible -- dans ce cas, affiche aussi le
    volume avec le supplement annonce ajoute, sans jamais remplacer le
    chiffre du modele par le chiffre ajuste."""
    volume_prevu = prevision["volume_prevu"]
    if volume_prevu is None:
        return kpi_card("Volume entrant (prévu)", "—", "dossiers, modèle Ensemble")

    def _fmt(v):
        return f"{v:,.0f}".replace(",", " ")

    volume_annonce = prevision.get("volume_annonce_jour") or 0
    if volume_annonce:
        return kpi_card_ajuste(
            "Volume entrant (prévu)", _fmt(volume_prevu), _fmt(prevision["volume_prevu_ajuste"]),
            f"ajusté : +{volume_annonce:g} dossiers annoncés",
        )
    return kpi_card("Volume entrant (prévu)", _fmt(volume_prevu), "dossiers, modèle Ensemble")


def construire_kpis(prevision):
    etp_global = prevision["etp_global"]
    charges_risque = prevision["charges_service_risque"]
    seuils_service = prevision["seuils_service"]

    services_alerte = [s for s in charges_risque.index if charges_risque[s] > seuils_service.get(s, float("inf"))]
    label_etp = "réel (contrats + congés)" if prevision["couverture_reelle"] else "estimé (taux d'absence historique)"

    return kpi_row(
        _carte_charge(prevision),
        kpi_card("Services à risque de surcharge", f"{len(services_alerte)} / {len(ORDRE_SERVICES)}",
                  ", ".join(services_alerte) if services_alerte else "aucun · si haut de fourchette",
                  statut="alerte" if services_alerte else "bon"),
        _carte_volume(prevision),
        kpi_card("ETP disponibles (prévu)", f"{etp_global:.1f}" if etp_global is not None else "—", label_etp),
    )


def _avec_dates_correctes(df_previsions, horizon_max):
    """La colonne 'date' de predictions_volumes_j30 est ancree sur la date
    reelle du jour ou le pipeline a tourne, pas sur le dernier jour reel
    des donnees (bug diagnostique avec l'utilisateur) -- on reconstruit la
    vraie date a partir de horizon_j (fiable, relatif au dernier jour reel)
    plutot que d'utiliser la colonne 'date' telle quelle."""
    prev = df_previsions[df_previsions["horizon_j"] <= horizon_max].sort_values("horizon_j").copy()
    prev["date"] = [PROCHAINS_JOURS_OUVRES[h - 1] for h in prev["horizon_j"]]
    return prev


def construire_fig_forecast(dj, df_previsions, horizon_j=30):
    fig = go.Figure()
    date_min = pd.Timestamp(TODAY) - timedelta(days=FENETRE_JOURS)
    hist = dj[dj["date"] >= date_min].sort_values("date")

    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["volume_entrant_jour"], name="Historique réel", mode="lines",
        line=dict(color="#3a372f", width=1.8), visible="legendonly",
        hovertemplate="%{x|%d/%m/%Y}<br>%{y:,.0f} dossiers<extra></extra>",
    ))

    df_previsions = df_previsions[df_previsions["horizon_j"] <= horizon_j]
    if len(df_previsions):
        prev = _avec_dates_correctes(df_previsions, horizon_j)
        fig.add_trace(go.Scatter(
            x=pd.concat([prev["date"], prev["date"][::-1]]),
            y=pd.concat([prev["volume_prevu_max"], prev["volume_prevu_min"][::-1]]),
            fill="toself", fillcolor="rgba(230,0,40,0.08)", line=dict(color="rgba(0,0,0,0)"),
            name="Intervalle", hoverinfo="skip", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=prev["date"], y=prev["volume_prevu"], name="Prévision", mode="lines",
            line=dict(color=SG_ROUGE, width=1.8, dash="dot"),
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:,.0f} dossiers (prévu)<extra></extra>",
        ))
    fig.update_yaxes(title_text="Dossiers / jour")
    return figure_layout_defaults(fig, hauteur=280)


def construire_table_previsions_service(djs, df_previsions, service=TOUS_SERVICES, horizon_j=HORIZON_TABLE_JOURS):
    """Prevision jour par jour (min / prevu / max). Pour un service donne,
    repartition illustrative de la prevision globale au prorata de sa part
    historique de volume ; pour "Tous services", sortie directe du modele
    (aucune repartition, donc aucune approximation)."""
    if df_previsions is None or not len(df_previsions):
        return html.P("Prévision indisponible.", className="table-note")
    prev = _avec_dates_correctes(df_previsions, horizon_j)
    if not len(prev):
        return html.P("Prévision indisponible.", className="table-note")

    if service != TOUS_SERVICES:
        part_volume = djs.groupby("Service")["volume_entrant_jour"].sum()
        part_volume = part_volume / part_volume.sum()
        part = part_volume.get(service, 0)
    else:
        part = 1

    lignes = []
    for _, row in prev.iterrows():
        jour = f"{NOM_JOURS[row['date'].weekday()][:3]} {row['date'].day:02d}/{row['date'].month:02d}"
        lignes.append({
            "Jour": jour,
            "Min": f"{row['volume_prevu_min'] * part:,.0f}".replace(",", " "),
            "Prévu": f"{row['volume_prevu'] * part:,.0f}".replace(",", " "),
            "Max": f"{row['volume_prevu_max'] * part:,.0f}".replace(",", " "),
        })

    return dash_table.DataTable(
        data=lignes,
        columns=[{"name": c, "id": c} for c in ["Jour", "Min", "Prévu", "Max"]],
        style_cell={"fontFamily": "-apple-system, sans-serif", "fontSize": "12.5px", "padding": "9px 12px",
                     "border": "none", "textAlign": "right", "fontVariantNumeric": "tabular-nums"},
        style_cell_conditional=[{"if": {"column_id": "Jour"}, "textAlign": "left", "minWidth": "110px"}],
        style_header={"fontWeight": "600", "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": "0.03em",
                       "backgroundColor": "#f6f4ee", "color": "#6b6759", "border": "none",
                       "borderBottom": "1px solid #e5e1d5"},
        style_data={"border": "none", "borderBottom": "1px solid #efece2", "color": "#3a372f"},
        style_data_conditional=[
            {"if": {"column_id": "Prévu"}, "fontWeight": "600", "color": "#14130f"},
        ],
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
    )


def caption_previsions_service(service):
    if service == TOUS_SERVICES:
        return "Sortie directe du modèle Ensemble (SARIMA + XGBoost)."
    return ("Répartition illustrative de la prévision globale selon la part historique "
            f"de volume de {service}.")


def bloc_charge_services(prevision):
    charges = prevision["charges_service"]
    charges_risque = prevision["charges_service_risque"]
    seuils = prevision["seuils_service"]
    if not len(charges):
        return html.P("Prévision indisponible.", className="table-note")
    max_echelle = max(charges.max(), seuils.reindex(charges.index).max()) * 1.15
    lignes = []
    for nom in ORDRE_SERVICES:
        if nom not in charges.index:
            continue
        # La hauteur de barre reste l'estimation centrale (informative) ;
        # le marquage "en alerte" utilise le scenario haut (cf.
        # construire_donnees_prevision), pour rester coherent avec le
        # KPI et la liste d'alertes juste a cote.
        en_alerte = charges_risque.get(nom, charges[nom]) > seuils.get(nom, float("inf"))
        lignes.append(service_bar_row(
            nom, charges[nom], seuils.get(nom, charges[nom]),
            max_echelle, SERVICES_COULEURS[nom], bool(en_alerte),
        ))
    return html.Div(lignes, className="service-bars")


def bloc_alertes(djs, prevision):
    charges_risque = prevision["charges_service_risque"]
    seuils = prevision["seuils_service"]
    services_alerte = sorted(
        (s for s in charges_risque.index if charges_risque[s] > seuils.get(s, float("inf"))),
        key=lambda s: charges_risque[s], reverse=True,
    )

    if not services_alerte:
        return html.P("Aucun service à risque pour ce jour, même en scénario haut.", className="table-note")

    # Le compteur "deja en alerte reelle" ne compare que le dernier jour
    # reel connu au jour cible : ca n'a de sens que pour J+1 (suite directe
    # du reel). Au-dela, rien n'a ete verifie sur les jours intermediaires
    # -- afficher "prevu de continuer" impliquerait a tort une continuite
    # ininterrompue jusqu'a un horizon lointain.
    horizon_j = prevision.get("horizon_j", 1)

    items = []
    for s in services_alerte:
        detail = f"jusqu'à {charges_risque[s]:.0f} dossiers/ETP si le haut de la fourchette se réalise"
        if horizon_j == 1:
            historique_service = djs[djs["Service"] == s].sort_values("date")
            recentes = historique_service.tail(14)
            jours_consecutifs = 0
            for v in recentes["alerte_surcharge"].values[::-1]:
                if v == 1:
                    jours_consecutifs += 1
                else:
                    break
            duree = f"{jours_consecutifs} j déjà + prévu de continuer" if jours_consecutifs > 0 else "nouveau risque prévu"
        else:
            duree = "risque isolé à cet horizon"
        items.append(alert_item(s, detail, duree))
    return html.Div(items, className="alert-list")


def options_jours_prevision():
    return [
        {"label": f"J+{h} · {NOM_JOURS[d.weekday()][:3]} {d.day:02d}/{d.month:02d}", "value": h}
        for h, d in enumerate(PROCHAINS_JOURS_OUVRES[:HORIZON_JOUR_MAX], start=1)
    ]


def bloc_grid_prevision(djs, prevision):
    d = prevision["date_cible"]
    jour = f"{NOM_JOURS[d.weekday()]} {d.day:02d}/{d.month:02d}/{d.year}"
    return html.Div([
        section_block(
            f"Charge par service — {jour}", None,
            html.Div([bloc_charge_services(prevision)], className="chart-block"),
        ),
        section_block(
            f"Alertes prévues — {jour}", "scénario haut",
            html.Div([bloc_alertes(djs, prevision)], className="chart-block"),
        ),
    ], className="grid-2")


def construire_table_risque_retard(df_risque_retard, service=TOUS_SERVICES, n=12):
    """Tâches "En cours" (créées dans les FENETRE_TACHES_EN_COURS_JOURS
    derniers jours) scorées en direct par le modèle de risque de retard
    (Bloc 3C) -- seule inférence ML live pour ce modèle, cf. loader.py.
    Triage réel : les tâches encore ouvertes, pas un backtest figé. Filtre
    service : sans lui, un service tres represente dans le top 12 global
    peut masquer completement le risque d'un autre service."""
    if not len(df_risque_retard):
        return html.P("Aucune tâche récente ouverte à évaluer.", className="table-note")

    a_risque = df_risque_retard[df_risque_retard["est_a_risque"] == 1]
    if service != TOUS_SERVICES:
        a_risque = a_risque[a_risque["Service"] == service]
    if not len(a_risque):
        return html.P("Aucune tâche à risque de retard détectée.", className="table-note")

    top = a_risque.sort_values("score_risque_retard", ascending=False).head(n).copy()
    top["jours_ouverte"] = (pd.Timestamp(TODAY) - top["Date_Creation"]).dt.days

    lignes = [{
        "Tâche": row.ID_Tache,
        "Agent": row.Matricule_Agent,
        "Service": row.Service,
        "Ouverte depuis": f"{row.jours_ouverte} j",
        "Risque": f"{row.score_risque_retard * 100:.0f}%",
    } for row in top.itertuples()]

    return dash_table.DataTable(
        data=lignes,
        columns=[{"name": c, "id": c} for c in ["Tâche", "Agent", "Service", "Ouverte depuis", "Risque"]],
        style_cell={"fontFamily": "-apple-system, sans-serif", "fontSize": "12.5px", "padding": "9px 12px",
                     "border": "none", "fontVariantNumeric": "tabular-nums"},
        style_cell_conditional=[
            {"if": {"column_id": "Ouverte depuis"}, "textAlign": "right"},
            {"if": {"column_id": "Risque"}, "textAlign": "right"},
        ],
        style_header={"fontWeight": "600", "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": "0.03em",
                       "backgroundColor": "#f6f4ee", "color": "#6b6759", "border": "none",
                       "borderBottom": "1px solid #e5e1d5"},
        style_data={"border": "none", "borderBottom": "1px solid #efece2", "color": "#3a372f"},
        style_data_conditional=[
            {"if": {"column_id": "Risque"}, "fontWeight": "600", "color": SG_ROUGE},
        ],
        style_as_list_view=True,
    )


def layout(dj, djs, df_previsions, m_charge, df_agents, df_absences, taux_repli, df_risque_retard=None):
    prevision = construire_donnees_prevision(dj, djs, df_previsions, m_charge, df_agents, df_absences, taux_repli, 1)

    filtre_jour = html.Div([
        html.Span("Prévision pour :", className="filtre-label"),
        dcc.Dropdown(
            id="surcharge-jour-prevision",
            options=options_jours_prevision(),
            value=1, clearable=False, style={"width": "220px"},
        ),
    ], className="filtre-row")

    contenu = [
        filtre_jour,
        html.Div(id="surcharge-kpi-row", children=construire_kpis(prevision)),

        html.Div(id="surcharge-prevision-grid", children=bloc_grid_prevision(djs, prevision)),

        section_block(
            "Prévision de volume", "Modèle Ensemble · MAE test 644,7 dossiers/j · à partir du prochain jour ouvré",
            html.Div([
                html.Span("Horizon :", className="filtre-label"),
                dcc.RadioItems(
                    id="surcharge-horizon-filter",
                    options=[{"label": " 7 jours", "value": 7}, {"label": " 30 jours", "value": 30}],
                    value=30, inline=True, style={"display": "flex", "gap": "16px"},
                ),
            ], className="filtre-row"),
            html.Div(id="surcharge-chart-forecast", children=chart_block(
                "", dcc.Graph(figure=construire_fig_forecast(dj, df_previsions, 30), config={"displayModeBar": False}),
                "Modèle Ensemble (SARIMA + XGBoost), erreur moyenne ≈ 645 dossiers/jour — détail sur la page Fiabilité.",
            )),
            html.Div([
                html.Span("Service :", className="filtre-label"),
                dcc.Dropdown(
                    id="surcharge-table-service-filter",
                    options=[{"label": TOUS_SERVICES, "value": TOUS_SERVICES}] +
                            [{"label": s, "value": s} for s in ORDRE_SERVICES],
                    value=TOUS_SERVICES, clearable=False, style={"width": "240px"},
                ),
            ], className="filtre-row"),
            html.Div(id="surcharge-table-previsions", children=chart_block(
                "Volume prévu, jour par jour (J+1 à J+7)",
                construire_table_previsions_service(djs, df_previsions, TOUS_SERVICES),
                caption_previsions_service(TOUS_SERVICES),
            )),
        ),

        section_block(
            "Tâches à risque de retard", None,
            html.Div([
                html.Span("Service :", className="filtre-label"),
                dcc.Dropdown(
                    id="surcharge-retard-service-filter",
                    options=[{"label": TOUS_SERVICES, "value": TOUS_SERVICES}] +
                            [{"label": s, "value": s} for s in ORDRE_SERVICES],
                    value=TOUS_SERVICES, clearable=False, style={"width": "240px"},
                ),
            ], className="filtre-row"),
            html.Div(id="surcharge-table-retard", children=chart_block(
                "",
                construire_table_risque_retard(df_risque_retard if df_risque_retard is not None else pd.DataFrame()),
                f"Tâches actuellement en cours, créées dans les {FENETRE_TACHES_EN_COURS_JOURS} derniers jours, "
                "scorées en direct par le modèle de risque de retard (Bloc 3C) — triée par risque décroissant.",
            )),
        ),
    ]
    return page_wrapper(*contenu)
