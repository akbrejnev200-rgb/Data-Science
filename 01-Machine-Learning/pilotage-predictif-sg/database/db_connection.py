"""
db_connection.py

Module partage pour la connexion a la base PostgreSQL.
Tous les autres scripts (setup_database, import_staging, data_cleaning,
feature_engineering, modelisation) importent get_engine() depuis ce module
plutot que de recopier la logique de connexion.
"""

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"


def get_engine():
    """Cree et retourne un engine SQLAlchemy connecte a pilotage_operationnel.

    Par defaut, charge .env (base locale). Pour cibler Neon (deploiement en
    ligne) sans toucher a .env, definir DEPLOY_TARGET=neon dans l'environnement
    avant de lancer le script -- charge alors .env.neon a la place.
    """
    env_name = ".env.neon" if os.getenv("DEPLOY_TARGET") == "neon" else ".env"
    env_path = BASE_DIR / env_name
    load_dotenv(dotenv_path=env_path, override=True)

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        database=os.getenv("DB_NAME"),
    )
    return create_engine(url)
