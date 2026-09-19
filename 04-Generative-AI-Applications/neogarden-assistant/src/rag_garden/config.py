"""Configuration centralisée : chemins, modèles, paramètres du RAG.

Les réglages non secrets vivent ici sous forme de constantes nommées. Le seul
secret (la clé API) est lu depuis l'environnement ou le fichier `.env`, jamais
écrit dans le code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Chemins -----------------------------------------------------------------
# Racine du projet : deux niveaux au-dessus de ce fichier (src/rag_garden/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Les documents sont encore à la racine. La Phase 2 les déplacera dans data/ ;
# cette ligne deviendra alors `PROJECT_ROOT / "data"`.
DATA_DIR = PROJECT_ROOT
# Index FAISS généré : artefact recalculable, ignoré par git.
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

DOCX_FILES = (
    "NeoGarden_FAQ.docx",
    "NeoGarden_CGU.docx",
    "NeoGarden_Politique_Retour.docx",
)
CATALOGUE_FILE = "Catalogue_Jardin_Amazon-2 - Catalogue.csv"

# --- Secret (jamais dans le code) ----------------------------------------------
# Chemin explicite vers .env : trouvé quel que soit le dossier de lancement.
load_dotenv(PROJECT_ROOT / ".env")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Modèles -----------------------------------------------------------------
EMBEDDING_MODEL = "dangvantuan/sentence-camembert-base"
# Modèle fixe (pas de routage automatique d'OpenRouter : il pourrait choisir un
# modèle non conversationnel et casser la démo).
LLM_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
LLM_BASE_URL = "https://openrouter.ai/api/v1"

# --- Découpage des documents et recherche ------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVER_K = 4  # nombre de passages renvoyés au LLM

# --- Robustesse des appels au LLM --------------------------------------------
LLM_MAX_RETRIES = 2
LLM_RETRY_DELAY_SECONDS = 2
