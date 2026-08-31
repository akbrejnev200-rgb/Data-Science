"""
auth.py

Gate d'authentification par mot de passe unique partage (prototype de
memoire, pas une infra bancaire de production -- pas de CSRF, pas de
rate-limiting, cookie de session non Secure car l'app tourne en HTTP local
sur localhost:8050). Voir app/generer_mot_de_passe.py pour choisir son
propre mot de passe sans jamais le transmettre en clair a un tiers.

init_auth(server) doit etre appele juste apres la creation de `server`,
avant l'assignation de `app.layout`.
"""

import os
import sys
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from flask import abort, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.security import check_password_hash

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))
from db_connection import MODELS_DIR

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# Prefixes accessibles sans etre connecte : le formulaire de login lui-meme,
# les bundles JS/CSS internes de Dash, le dossier assets/ (CSS de l'app,
# non sensible) et la favicon. Tout le reste (pages, /_dash-update-component,
# /model-assets/...) est protege.
PREFIXES_PUBLICS = ("/login", "/_dash-component-suites", "/assets", "/_favicon.ico")


def init_auth(server):
    server.secret_key = os.environ["FLASK_SECRET_KEY"]
    server.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=8)

    @server.before_request
    def exiger_connexion():
        if request.path.startswith(PREFIXES_PUBLICS):
            return None
        if session.get("authenticated"):
            return None
        if request.path.startswith("/_dash-"):
            # Appel XHR interne de Dash : une redirection HTML casserait le
            # fetch() cote client qui attend du JSON. On echoue proprement.
            abort(401)
        return redirect(url_for("login_page"))

    @server.route("/login", methods=["GET", "POST"])
    def login_page():
        erreur = None
        if request.method == "POST":
            identifiant = request.form.get("username", "")
            mot_de_passe = request.form.get("password", "")
            identifiant_ok = identifiant == os.environ["APP_USERNAME"]
            mot_de_passe_ok = check_password_hash(os.environ["APP_PASSWORD_HASH"], mot_de_passe)
            if identifiant_ok and mot_de_passe_ok:
                session["authenticated"] = True
                session["username"] = identifiant
                session.permanent = True
                return redirect("/")
            erreur = "Identifiant ou mot de passe incorrect."
        return render_template("login.html", erreur=erreur)

    @server.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login_page"))

    @server.route("/model-assets/<path:filename>")
    def model_assets(filename):
        # Route protegee par exiger_connexion() ci-dessus (ne commence pas par
        # un prefixe public) : sert par ex. shap_anomalies_summary.png
        # uniquement aux utilisateurs connectes.
        return send_from_directory(MODELS_DIR, filename)
