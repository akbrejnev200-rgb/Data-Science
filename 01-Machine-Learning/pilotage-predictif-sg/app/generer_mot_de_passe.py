"""
generer_mot_de_passe.py

A executer manuellement UNE FOIS pour configurer le login de l'application.
Ne transmet jamais le mot de passe en clair : imprime seulement les deux
lignes a coller dans le fichier .env a la racine du projet.

Usage :
    python generer_mot_de_passe.py
"""

import secrets
from getpass import getpass

from werkzeug.security import generate_password_hash

if __name__ == "__main__":
    identifiant = input("Choisis l'identifiant de connexion : ").strip()
    mot_de_passe = getpass("Choisis le mot de passe de l'application : ")
    confirmation = getpass("Confirme le mot de passe : ")
    if mot_de_passe != confirmation:
        print("Les deux saisies ne correspondent pas. Relance le script.")
        raise SystemExit(1)

    secret_key = secrets.token_hex(32)
    password_hash = generate_password_hash(mot_de_passe)

    print("\nAjoute ces trois lignes dans le fichier .env a la racine du projet :\n")
    print(f"FLASK_SECRET_KEY={secret_key}")
    print(f"APP_USERNAME={identifiant}")
    print(f"APP_PASSWORD_HASH={password_hash}")
