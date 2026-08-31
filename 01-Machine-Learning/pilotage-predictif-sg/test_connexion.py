from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

print(f"Tentative de connexion à {DB_HOST}:{DB_PORT}/{DB_NAME} avec l'utilisateur {DB_USER}...")

try:
    engine = create_engine(
        f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}',
        connect_args={"connect_timeout": 10}
    )
    with engine.connect() as conn:
        result = conn.execute(text('SELECT version()'))
        print('Connexion réussie :', result.fetchone()[0])
except Exception as e:
    print('Erreur de connexion :', str(e))
