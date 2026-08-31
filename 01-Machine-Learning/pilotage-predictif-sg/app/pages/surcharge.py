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

    # Alertes basees sur la meme prevision centrale que le KPI "Volume
    # entrant (prevu)" et les barres du graphique -- un seul chiffre de
    # volume dans toute la page, pas de scenario haut separe qui ne
    # correspondrait a aucun autre nombre affiche.
    charges_service = _repartir(volume_prevu)

    return {
        "date_cible": jour["date"],
        "horizon_j": horizon_j,
        "volume_prevu": volume_prevu,
        "etp_global": jour["etp_global"],
        "etp_service": etp_service,
        "couverture_reelle": jour["couverture_reelle"],
        "charge_prevue": jour["charge_prevue"],
        "charges_service": charges_service,
        "seuils_service": seuils_service,
        "nb_etp_impactes_jour": jour["nb_etp_impactes_jour"],
        "volume_annonce_jour": jour["volume_annonce_jour"],
        "volume_prevu_ajuste": jour["volume_prevu_ajuste"],
        "charge_prevue_ajustee": jour["charge_prevue_ajustee"],
    }


def _carte_taches_risque(df_risque_retard):
    """Nombre de taches actuellement "En cours" signalees a risque de retard
    (Bloc 3C, scoring en direct) -- resume en un chiffre la table "Taches a
    risque de retard" plus bas sur la meme page, sans nouveau calcul.
    Remplace l'ancien KPI "Charge globale/ETP", redondant avec le graphique
    "Charge par service" juste en dessous (meme information, deja par
    service)."""
    if df_risque_retard is None or not len(df_risque_retard):
        return kpi_card("Tâches à risque de retard", "—", "aucune tâche en cours évaluée")
    total = len(df_risque_retard)
    a_risque = int(df_risque_retard["est_a_risque"].sum())
    return kpi_card(
        "Tâches à risque de retard", f"{a_risque}", f"sur {total} tâches en cours évaluées",
        statut="alerte" if a_risque else "bon",
    )


def construire_risque_surcharge_fenetre(dj, djs, df_previsions, m_charge, df_agents, df_absences, taux_repli, horizon_depart=1, horizon_jours=14):
    """Services dont la charge par ETP PREVUE (estimation centrale) est EN
    MOYENNE, sur une fenetre de `horizon_jours` jours ouvres DEMARRANT a
    J+`horizon_depart`, au-dessus du seuil historique (P75) -- la fenetre
    est ancree sur le jour actuellement selectionne dans "Prevision pour"
    (horizon_depart), pas toujours sur aujourd'hui : inspecter J+5 montre
    les 2 semaines qui suivent J+5, pas celles qui suivent aujourd'hui.

    Pourquoi une moyenne et pas "au moins un jour depasse le seuil" (version
    initiale) : un seuil P75 est par definition deja depasse par 25% des
    jours pris individuellement. Sur une fenetre de 14 jours, la probabilite
    qu'AU MOINS UN jour le depasse avoisine 1-0.75**14 ~= 98%, meme sans
    aucune vraie tendance a la hausse -- ca revient a alerter (presque)
    tout le temps, pour (presque) tous les services, sans rien dire de reel
    (constate en pratique : 4/4 services systematiquement signales). Une
    moyenne sur la fenetre lisse le bruit quotidien : elle ne depasse le
    seuil que si la pression est reellement soutenue sur la periode, pas a
    cause d'un pic isole.

    construire_serie_prevision doit recevoir la chaine complete depuis J+1
    (necessaire pour la recursion des lags et l'alignement de son compteur
    d'horizon interne avec df_previsions.horizon_j) ; seule la fin de la
    serie (les horizon_jours derniers jours, a partir de horizon_depart) est
    utilisee pour la moyenne -- au-dela de J+30 (fin de la prevision
    modele), la fenetre se retrecit naturellement plutot que de planter.

    Retourne (moyennes_en_alerte, date_debut, date_fin) : moyennes_en_alerte
    ne contient que les services en depassement ; date_debut/date_fin sont
    les bornes reelles (calendaires) de la fenetre de horizon_jours jours
    OUVRES -- a afficher telles quelles plutot qu'un texte du type "2
    semaines" qui suggere a tort une duree calendaire fixe (une fenetre de
    14 jours ouvres peut chevaucher 2 week-ends et donc couvrir 18 jours
    calendaires reels)."""
    fin = horizon_depart + horizon_jours - 1
    jours_cibles = PROCHAINS_JOURS_OUVRES[:fin]
    serie = construire_serie_prevision(dj, df_previsions, df_agents, df_absences, taux_repli, m_charge, jours_cibles)
    fenetre = serie[horizon_depart - 1:]
    date_debut = fenetre[0]["date"] if fenetre else None
    date_fin = fenetre[-1]["date"] if fenetre else None

    seuils_service = djs.groupby("Service")["charge_par_etp"].quantile(0.75)
    part_volume = djs.groupby("Service")["volume_entrant_jour"].sum()
    part_volume = part_volume / part_volume.sum()

    charges_par_service = {s: [] for s in ORDRE_SERVICES}
    for jour in fenetre:
        if jour["volume_prevu"] is None:
            continue
        etp_service = jour["etp_service"]
        for s in ORDRE_SERVICES:
            if s not in etp_service.index or etp_service[s] <= 0:
                continue
            charges_par_service[s].append((jour["volume_prevu"] * part_volume.get(s, 0)) / etp_service[s])

    moyennes_en_alerte = {}
    for s, valeurs in charges_par_service.items():
        if not valeurs:
            continue
        moyenne = sum(valeurs) / len(valeurs)
        if moyenne > seuils_service.get(s, float("inf")):
            moyennes_en_alerte[s] = moyenne

    return moyennes_en_alerte, date_debut, date_fin


def _carte_volume(prevision):
    """Volume entrant : sortie brute du modele (Ensemble), sauf si une
    annonce est active pour ce jour cible -- dans ce cas, affiche aussi le
    volume avec le supplement annonce ajoute, sans jamais remplacer le
    chiffre du modele par le chiffre ajuste."""
    volume_prevu = prevision["volume_prevu"]
    if volume_prevu is None:
        return kpi_card("Volume entrant (prévu)", "—")

    def _fmt(v):
        return f"{v:,.0f}".replace(",", " ")

    volume_annonce = prevision.get("volume_annonce_jour") or 0
    if volume_annonce:
        return kpi_card_ajuste(
            "Volume entrant (prévu)", _fmt(volume_prevu), _fmt(prevision["volume_prevu_ajuste"]),
            f"ajusté : +{volume_annonce:g} dossiers annoncés",
        )
    return kpi_card("Volume entrant (prévu)", _fmt(volume_prevu))


def construire_kpis(prevision, risque_fenetre, df_risque_retard):
    etp_global = prevision["etp_global"]
    label_etp = "réel (contrats + congés)" if prevision["couverture_reelle"] else "estimé (taux d'absence historique)"

    moyennes_en_alerte, date_debut_fenetre, date_fin_fenetre = risque_fenetre
    services_alerte = sorted(moyennes_en_alerte, key=moyennes_en_alerte.get, reverse=True)

    # Plage de dates reelle affichee telle quelle (pas "2 semaines", qui
    # suggere a tort une duree calendaire fixe -- 14 jours OUVRES peuvent
    # chevaucher 2 week-ends et couvrir jusqu'a 18 jours calendaires).
    def _fmt_date(d):
        return f"{d.day:02d}/{d.month:02d}" if d is not None else "?"
    periode = f"{_fmt_date(date_debut_fenetre)} → {_fmt_date(date_fin_fenetre)}"

    # Titre volontairement court pour tenir sur une seule ligne dans la
    # carte KPI -- le detail precis (duree + plage de dates reelle) va
    # dans le sous-texte, qui peut s'étaler sur plusieurs lignes sans que
    # ça ait l'air casse (meme traitement que les autres cartes).
    detail_fenetre = f"14 j. ouvrés, {periode}"
    sous_texte = f"{', '.join(services_alerte)} · {detail_fenetre}" if services_alerte else f"aucun · {detail_fenetre}"

    return kpi_row(
        _carte_taches_risque(df_risque_retard),
        kpi_card("Services en surcharge soutenue", f"{len(services_alerte)} / {len(ORDRE_SERVICES)}",
                  sous_texte, statut="alerte" if services_alerte else "bon"),
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
    seuils = prevision["seuils_service"]
    if not len(charges):
        return html.P("Prévision indisponible.", className="table-note")
    max_echelle = max(charges.max(), seuils.reindex(charges.index).max()) * 1.15
    lignes = []
    for nom in ORDRE_SERVICES:
        if nom not in charges.index:
            continue
        # Le marquage "en alerte" compare la meme estimation centrale que
        # la hauteur de barre au seuil -- coherent avec le KPI et la liste
        # d'alertes juste a cote (un seul chiffre de volume sur la page).
        en_alerte = charges[nom] > seuils.get(nom, float("inf"))
        lignes.append(service_bar_row(
            nom, charges[nom], seuils.get(nom, charges[nom]),
            max_echelle, SERVICES_COULEURS[nom], bool(en_alerte),
        ))
    return html.Div(lignes, className="service-bars")


def bloc_alertes(djs, prevision):
    charges = prevision["charges_service"]
    seuils = prevision["seuils_service"]
    services_alerte = sorted(
        (s for s in charges.index if charges[s] > seuils.get(s, float("inf"))),
        key=lambda s: charges[s], reverse=True,
    )

    if not services_alerte:
        return html.P("Aucun service à risque pour ce jour, sur la base de la prévision.", className="table-note")

    # Le compteur "deja en alerte reelle" ne compare que le dernier jour
    # reel connu au jour cible : ca n'a de sens que pour J+1 (suite directe
    # du reel). Au-dela, rien n'a ete verifie sur les jours intermediaires
    # -- afficher "prevu de continuer" impliquerait a tort une continuite
    # ininterrompue jusqu'a un horizon lointain.
    horizon_j = prevision.get("horizon_j", 1)

    items = []
    for s in services_alerte:
        detail = f"≈ {charges[s]:.0f} dossiers/ETP prévus, au-dessus du seuil habituel ({seuils[s]:.0f})"
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
            f"Alertes prévues — {jour}", "sur la prévision",
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
    risque_fenetre = construire_risque_surcharge_fenetre(dj, djs, df_previsions, m_charge, df_agents, df_absences, taux_repli, 1)
    # risque_fenetre = (moyennes_en_alerte, date_debut, date_fin)

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
        html.Div(id="surcharge-kpi-row", children=construire_kpis(prevision, risque_fenetre, df_risque_retard)),

        html.Div(id="surcharge-prevision-grid", children=bloc_grid_prevision(djs, prevision)),

        section_block(
            "Prévision de volume", None,
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
                "Modèle Ensemble (SARIMA + XGBoost), erreur moyenne ≈ 647 dossiers/jour — détail sur la page Fiabilité.",
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
