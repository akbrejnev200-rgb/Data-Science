"""Interface Streamlit de l'assistant NeoGarden : affichage uniquement.

Toute la logique (recherche, prompts, appel au LLM, erreurs) vit dans le paquet
`rag_garden`. Ici : la mise en page, l'état de la session et les messages.
"""

import logging
import os

import streamlit as st
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from rag_garden import config
from rag_garden.errors import KnowledgeBaseError, LLMRequestError, LLMUnavailableError
from rag_garden.memory import to_langchain_messages
from rag_garden.pipeline import Answer, answer_question, build_rag_chain, load_retriever

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Émoji « agriculteur » : 🧑 + liant invisible (U+200D) + 🌾
AVATARS = {"user": "\U0001f9d1‍\U0001f33e", "assistant": "🌿"}
WELCOME_MESSAGE = (
    "Bonjour ! Je suis l'assistant NeoGarden. Comment puis-je vous aider pour "
    "vos achats de jardinage ou le suivi de vos commandes ?"
)
FRIENDLY_ERROR = (
    "Désolé, je rencontre un problème technique momentané. "
    "Merci de réessayer votre question dans quelques instants."
)


# --- Ressources mises en cache (calculées une seule fois par session serveur) ---
@st.cache_resource(show_spinner=False)
def get_retriever():
    """Base de connaissances : partie lourde, indépendante de la clé API."""
    return load_retriever()


@st.cache_resource(show_spinner=False)
def get_chain(api_key: str) -> Runnable:
    """Chaîne RAG : partie légère, refaite si la clé change (sans recalculer l'index)."""
    return build_rag_chain(get_retriever(), api_key)


# --- Blocs d'affichage ---
def render_sidebar(api_key: str | None) -> None:
    """Barre latérale : état de la clé API et modèle utilisé."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        if api_key:
            st.success("Clé API OpenRouter chargée depuis .env")
        else:
            st.error(
                "Aucune clé API trouvée. Ajoute OPENROUTER_API_KEY dans le fichier .env"
            )
            st.markdown("Obtenez votre clé sur [openrouter.ai/keys](https://openrouter.ai/keys)")
        st.markdown("---")
        model_name = config.LLM_MODEL.split(":")[0]
        st.markdown(
            f"Cet assistant utilise un système RAG couplé au modèle `{model_name}` "
            "via OpenRouter."
        )


def display_sources(sources: list[Document] | None) -> None:
    """Affiche les passages utilisés sous la réponse, pour prouver l'ancrage."""
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)} extraits utilisés)"):
        for i, doc in enumerate(sources, start=1):
            source_name = os.path.basename(
                doc.metadata.get("source", "Catalogue produits (CSV)")
            )
            st.markdown(f"**{i}. {source_name}**")
            st.caption(doc.page_content)


def render_history() -> None:
    """Rejoue tous les messages de la conversation."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=AVATARS.get(message["role"])):
            st.markdown(message["content"])
            display_sources(message.get("sources"))


# --- État et logique d'interaction ---
def init_session_state() -> None:
    """Crée la conversation avec le message de bienvenue, au premier passage."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]


def init_chain(api_key: str | None) -> Runnable | None:
    """Prépare la chaîne RAG ; affiche le problème à l'utilisateur si elle échoue."""
    if not api_key:
        st.warning(
            "👈 Veuillez configurer la clé API OpenRouter (fichier .env) "
            "pour activer le chatbot."
        )
        return None
    with st.spinner("Chargement de la base de connaissances NeoGarden..."):
        try:
            return get_chain(api_key)
        except KnowledgeBaseError as exc:
            logger.error("Base de connaissances inutilisable : %s", exc)
            st.error(f"Erreur d'initialisation : {exc}")
        except Exception as exc:
            # Dernier filet, à la frontière de l'application : on journalise la
            # trace complète mais l'utilisateur ne voit qu'un message lisible.
            logger.exception("Erreur d'initialisation inattendue")
            st.error(f"Erreur d'initialisation : {exc}")
    return None


def generate_answer(chain: Runnable, question: str) -> Answer | None:
    """Interroge le RAG. Renvoie None si la réponse n'a pas pu être produite."""
    # On exclut le message de bienvenue (index 0) et la question actuelle (dernier).
    history = to_langchain_messages(st.session_state.messages[1:-1])
    try:
        return answer_question(chain, question, history)
    except (LLMUnavailableError, LLMRequestError) as exc:
        logger.error("Appel au LLM en échec : %s", exc)
    except Exception:
        # Dernier filet, à la frontière de l'application (voir init_chain).
        logger.exception("Erreur inattendue pendant la génération de la réponse")
    return None


def handle_user_input(chain: Runnable | None) -> None:
    """Traite la question saisie : l'affiche, obtient la réponse, la mémorise."""
    prompt = st.chat_input(
        "Tapez votre message ici (ex: Livrez-vous en Belgique ?)..."
    )
    if not prompt:
        return
    if chain is None:
        st.error("Veuillez d'abord configurer la clé API OpenRouter.")
        return

    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Je cherche..."):
            answer = generate_answer(chain, prompt)
        if answer is None:
            st.error(FRIENDLY_ERROR)
            message = {"role": "assistant", "content": FRIENDLY_ERROR}
        else:
            st.markdown(answer.text)
            display_sources(answer.sources)
            message = {
                "role": "assistant",
                "content": answer.text,
                "sources": answer.sources,
            }
        st.session_state.messages.append(message)


def main() -> None:
    """Assemble la page."""
    st.set_page_config(page_title="NeoGarden Assistant", page_icon="🌿", layout="centered")
    st.title("🌿 Assistant NeoGarden")
    st.markdown(
        "Posez vos questions sur nos produits, vos commandes ou nos conditions de retour !"
    )
    api_key = config.OPENROUTER_API_KEY
    render_sidebar(api_key)
    init_session_state()
    chain = init_chain(api_key)
    render_history()
    handle_user_input(chain)
    st.divider()
    st.caption(
        "🌿 NeoGarden est une boutique fictive — démo d'un système RAG "
        "(Retrieval-Augmented Generation)."
    )


main()
