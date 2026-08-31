"""callbacks/cb_incidents.py

Trois callbacks independants :
- basculer_type : bascule l'affichage entre le formulaire "Incident ETP" et
  "Annonce" (dcc.Store memorise le type actif, lu par le callback de
  soumission).
- soumettre : valide puis ecrit le nouvel evenement dans incidents_manager
  (data.loader.inserer_incident), recharge l'historique et vide le
  formulaire. Pour une annonce, la date de fin saisie n'est jamais stockee
  telle quelle -- seule duree_jours = (date_fin - date_debut + 1) l'est,
  la table incidents_manager n'ayant qu'une colonne date_evenement (debut).
- supprimer_ligne : la croix de suppression de dash_table (row_deletable)
  retire deja la ligne cote client ; ce callback se contente de repercuter
  la suppression en base (data.loader.supprimer_incident) en comparant
  data/data_previous.
"""

from datetime import datetime

from dash import Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from components.layout_common import chart_block
from data.loader import charger_incidents, inserer_incident, supprimer_incident
from pages.incidents import construire_table_historique

NB_CHAMPS_A_VIDER = 11  # cf. CHAMPS_FORMULAIRE dans pages/incidents.py


def _feedback(message, ok=True):
    return html.Div(message, className=f"form-feedback {'success' if ok else 'error'}")


def register(app):
    @app.callback(
        Output("incidents-type-store", "data"),
        Output("incidents-btn-etp", "className"),
        Output("incidents-btn-annonce", "className"),
        Output("incidents-form-etp", "style"),
        Output("incidents-form-annonce", "style"),
        Input("incidents-btn-etp", "n_clicks"),
        Input("incidents-btn-annonce", "n_clicks"),
        prevent_initial_call=True,
    )
    def basculer_type(_n_etp, _n_annonce):
        type_actif = "annonce" if ctx.triggered_id == "incidents-btn-annonce" else "incident_etp"
        classe_base = "type-toggle-btn"
        classe_active = "type-toggle-btn type-toggle-btn-active"
        if type_actif == "incident_etp":
            return (type_actif, classe_active, classe_base,
                    {"display": "grid"}, {"display": "none"})
        return (type_actif, classe_base, classe_active,
                {"display": "none"}, {"display": "grid"})

    @app.callback(
        Output("incidents-history-block", "children", allow_duplicate=True),
        Output("incidents-feedback", "children", allow_duplicate=True),
        Output("incidents-date-etp", "date"),
        Output("incidents-service-etp", "value"),
        Output("incidents-nb-etp", "value"),
        Output("incidents-motif-etp", "value"),
        Output("incidents-duree-etp", "value"),
        Output("incidents-date-debut-annonce", "date"),
        Output("incidents-date-fin-annonce", "date"),
        Output("incidents-service-annonce", "value"),
        Output("incidents-motif-annonce", "value"),
        Output("incidents-volume-annonce", "value"),
        Output("incidents-commentaire-annonce", "value"),
        Input("incidents-submit", "n_clicks"),
        State("incidents-type-store", "data"),
        State("incidents-date-etp", "date"),
        State("incidents-service-etp", "value"),
        State("incidents-nb-etp", "value"),
        State("incidents-motif-etp", "value"),
        State("incidents-duree-etp", "value"),
        State("incidents-date-debut-annonce", "date"),
        State("incidents-date-fin-annonce", "date"),
        State("incidents-service-annonce", "value"),
        State("incidents-motif-annonce", "value"),
        State("incidents-volume-annonce", "value"),
        State("incidents-commentaire-annonce", "value"),
        prevent_initial_call=True,
    )
    def soumettre(_n_clicks, type_evt,
                  date_etp, service_etp, nb_etp, motif_etp, duree_etp,
                  date_debut, date_fin, service_annonce, motif_annonce, volume_annonce, commentaire):
        garder_champs = (no_update,) * NB_CHAMPS_A_VIDER

        if type_evt == "incident_etp":
            if not all([date_etp, service_etp, motif_etp]) or nb_etp is None or duree_etp is None:
                return (no_update, _feedback("Merci de renseigner tous les champs de l'incident ETP.", ok=False),
                        *garder_champs)
            if nb_etp < 0 or duree_etp <= 0:
                return (no_update, _feedback("ETP impactés et durée doivent être positifs.", ok=False),
                        *garder_champs)
            inserer_incident(
                date_evenement=datetime.fromisoformat(date_etp).date(), service=service_etp,
                type_evenement="incident_etp",
                motif=motif_etp, nb_etp_impactes=nb_etp, duree_jours=duree_etp,
                volume_supplementaire_estime=None, commentaire=None,
            )
            message = f"Incident ETP enregistré pour {service_etp}."

        elif type_evt == "annonce":
            if not all([date_debut, date_fin, service_annonce, motif_annonce]) or volume_annonce is None:
                return (no_update, _feedback("Merci de renseigner tous les champs de l'annonce.", ok=False),
                        *garder_champs)
            debut = datetime.fromisoformat(date_debut).date()
            fin = datetime.fromisoformat(date_fin).date()
            if fin < debut:
                return (no_update, _feedback("La date de fin doit être postérieure à la date de début.", ok=False),
                        *garder_champs)
            if volume_annonce < 0:
                return (no_update, _feedback("Le volume supplémentaire estimé doit être positif.", ok=False),
                        *garder_champs)
            duree_jours = (fin - debut).days + 1
            inserer_incident(
                date_evenement=debut, service=service_annonce, type_evenement="annonce",
                motif=motif_annonce, nb_etp_impactes=None, duree_jours=duree_jours,
                volume_supplementaire_estime=volume_annonce, commentaire=commentaire,
            )
            message = f"Annonce enregistrée pour {service_annonce} ({duree_jours} j)."

        else:
            raise PreventUpdate

        nouveau_bloc = chart_block(
            "Historique des 30 derniers événements",
            construire_table_historique(charger_incidents(30)),
            "Cliquer sur la croix en début de ligne pour supprimer un événement (définitif).",
        )
        champs_vides = (None,) * NB_CHAMPS_A_VIDER
        return (nouveau_bloc, _feedback(message, ok=True), *champs_vides)

    @app.callback(
        Output("incidents-feedback", "children", allow_duplicate=True),
        Input("incidents-history-table", "data"),
        State("incidents-history-table", "data_previous"),
        prevent_initial_call=True,
    )
    def supprimer_ligne(data, data_previous):
        if data_previous is None or len(data) >= len(data_previous):
            raise PreventUpdate
        ids_avant = {ligne["id"] for ligne in data_previous}
        ids_apres = {ligne["id"] for ligne in data}
        supprimes = ids_avant - ids_apres
        for id_incident in supprimes:
            supprimer_incident(id_incident)
        return _feedback(f"{len(supprimes)} événement(s) supprimé(s).", ok=True)
