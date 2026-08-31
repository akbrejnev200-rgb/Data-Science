"""
layout_common.py

Coquille de l'application (sidebar + barre superieure) et blocs
reutilises par les 4 pages : cartes KPI, blocs graphique, barres de charge
par service, liste d'alertes. Style aligne sur la maquette validee : sobre,
tabulaire, rouge SG reserve aux alertes.
"""

from dash import dcc, html

PAGES = [
    ("surcharge", "Surcharge & prévisions", "Vue d'ensemble"),
    ("reallocation", "Réallocation équitable", "Vue d'ensemble"),
    ("derive", "Dérive individuelle", "Qualité & contrôle"),
    ("fiabilite", "Fiabilité du modèle", "Qualité & contrôle"),
    ("incidents", "Incidents & Annonces", "Saisie manuelle"),
]


def sidebar(pathname):
    def actif(page):
        return pathname == f"/{page}" or (page == "surcharge" and pathname == "/")

    sections = []
    derniere_section = None
    liens_section = []
    for page, label, section in PAGES:
        if section != derniere_section:
            if liens_section:
                sections.append(html.Div(liens_section, className="nav-container"))
            sections.append(html.Div(section, className="nav-section-label"))
            derniere_section = section
            liens_section = []
        liens_section.append(dcc.Link(
            label, href=f"/{page}",
            className="nav-link nav-link-active" if actif(page) else "nav-link",
            refresh=False,
        ))
    if liens_section:
        sections.append(html.Div(liens_section, className="nav-container"))

    return html.Div([
        html.Div([
            html.Div([
                html.Div([html.Div(className="bar"), html.Div("Société Générale", className="brand-name")],
                         className="brand-mark"),
                html.Div("Pilotage Prédictif", className="brand-title"),
                html.Div("Opérations Personnes Physiques", className="brand-sub"),
            ], className="brand"),
            *sections,
        ], className="sidebar-scroll"),
        html.Div(html.A("↩ Se déconnecter", href="/logout"), className="sidebar-foot"),
    ], className="sidebar")


def topbar(titre, sous_titre):
    return html.Div([
        html.Div([
            html.Div(titre, className="topbar-title"),
            html.Div(sous_titre, className="topbar-date"),
        ]),
        html.Div([html.Span(className="dot"), "Synchronisé avec PostgreSQL"], className="last-sync"),
    ], className="topbar")


def kpi_card(label, valeur, sous_texte=None, statut="neutre"):
    enfants = [
        html.Div(label, className="kpi-label"),
        html.Div(str(valeur), className="kpi-value"),
    ]
    if sous_texte:
        enfants.append(html.Div(
            [html.Span(className="dot"), sous_texte],
            className=f"kpi-subtext statut-{statut}",
        ))
    return html.Div(enfants, className="kpi-card")


def kpi_card_ajuste(label, valeur_modele, valeur_ajustee, sous_texte=None, statut="avertissement"):
    """Variante de kpi_card() pour une valeur ajustee en direct par un
    evenement incidents_manager (cf. data.loader.construire_serie_prevision) :
    affiche toujours le chiffre du modele ET le chiffre ajuste, jamais l'un
    a la place de l'autre -- pour qu'on puisse toujours distinguer ce que
    dit la statistique de ce que la saisie manuelle y ajoute. N'est utilisee
    que lorsqu'un ajustement est reellement actif ; sinon la page retombe
    sur kpi_card() (un seul chiffre)."""
    enfants = [
        html.Div(label, className="kpi-label"),
        html.Div([
            html.Span(str(valeur_modele), className="kpi-value kpi-value-base"),
            html.Span("→", className="kpi-value-arrow"),
            html.Span(str(valeur_ajustee), className="kpi-value kpi-value-adjusted"),
        ], className="kpi-value-row"),
    ]
    if sous_texte:
        enfants.append(html.Div(
            [html.Span(className="dot"), sous_texte],
            className=f"kpi-subtext statut-{statut}",
        ))
    return html.Div(enfants, className="kpi-card")


def kpi_row(*cartes):
    return html.Div(list(cartes), className="kpi-row")


def section_block(titre, meta, *contenu):
    return html.Div([
        html.Div([
            html.H3(titre, className="section-title"),
            html.Span(meta, className="section-meta") if meta else None,
        ], className="section-head"),
        *contenu,
    ], className="section-block")


def chart_block(titre, graphique, caption=None):
    enfants = [html.H3(titre, className="chart-title"), graphique]
    if caption:
        enfants.append(html.P(caption, className="chart-caption"))
    return html.Div(enfants, className="chart-block")


def service_bar_row(nom, valeur, seuil, max_echelle, couleur, en_alerte):
    largeur_pct = min(100, valeur / max_echelle * 100)
    seuil_pct = min(100, seuil / max_echelle * 100)
    return html.Div([
        html.Div(nom, className="service-name"),
        html.Div([
            html.Div(className="service-fill", style={"width": f"{largeur_pct}%", "background": couleur}),
            html.Div(className="service-threshold", style={"left": f"{seuil_pct}%"}),
        ], className="service-track"),
        html.Div(f"{valeur:.0f}", className="service-val over" if en_alerte else "service-val"),
    ], className="service-row")


def alert_item(service, detail, duree, severe=True):
    return html.Div([
        html.Div([
            html.Div(className="sev-stripe" if severe else "sev-stripe warn"),
            html.Div([
                html.Div(service, className="alert-service"),
                html.Div(detail, className="alert-detail"),
            ]),
        ], className="alert-left"),
        html.Div(duree, className="alert-days"),
    ], className="alert-item")


def page_wrapper(*contenu):
    return html.Div(list(contenu), className="content")


def figure_layout_defaults(fig, hauteur=280):
    fig.update_layout(
        height=hauteur,
        margin=dict(l=46, r=16, t=34, b=36),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif", color="#14130f", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#6b6759", size=11)),
        xaxis=dict(gridcolor="#efece2", linecolor="#e5e1d5", tickfont=dict(color="#948f80", size=11)),
        yaxis=dict(gridcolor="#efece2", linecolor="#e5e1d5", tickfont=dict(color="#948f80", size=11)),
    )
    return fig
