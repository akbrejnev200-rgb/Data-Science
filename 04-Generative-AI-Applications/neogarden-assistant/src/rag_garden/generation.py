"""Génération de la réponse : LLM (via OpenRouter), chaîne de réponse, retry."""

import logging
import time

import openai
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from rag_garden import config
from rag_garden.errors import LLMRequestError, LLMUnavailableError
from rag_garden.prompts import build_qa_prompt

logger = logging.getLogger(__name__)

# Erreurs TEMPORAIRES : réseau coupé, trop de requêtes (429), fournisseur
# surchargé (5xx, ex. 502). Réessayer a du sens. Toute autre erreur du SDK
# (clé invalide, requête mal formée...) est définitive : on ne réessaie pas.
TRANSIENT_ERRORS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def build_llm(api_key: str) -> ChatOpenAI:
    """Client du LLM via OpenRouter (endpoint compatible OpenAI)."""
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=api_key,
        base_url=config.LLM_BASE_URL,
    )


def build_answer_chain(llm: ChatOpenAI) -> Runnable:
    """Chaîne qui écrit la réponse à partir des passages retrouvés."""
    return create_stuff_documents_chain(llm, build_qa_prompt())


def invoke_with_retry(
    chain: Runnable,
    payload: dict,
    max_retries: int = config.LLM_MAX_RETRIES,
    delay: float = config.LLM_RETRY_DELAY_SECONDS,
) -> dict:
    """Appelle la chaîne en réessayant seulement sur les erreurs temporaires.

    Lève `LLMUnavailableError` si l'erreur temporaire persiste après les
    tentatives, `LLMRequestError` si le fournisseur refuse la requête. Les
    autres exceptions (bugs, erreurs inattendues) ne sont pas masquées.
    """
    attempts = max_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            return chain.invoke(payload)
        except TRANSIENT_ERRORS as exc:
            logger.warning(
                "Tentative %d/%d échouée (erreur temporaire) : %s", attempt, attempts, exc
            )
            if attempt == attempts:
                raise LLMUnavailableError(
                    "Le fournisseur du LLM est indisponible après plusieurs tentatives."
                ) from exc
            time.sleep(delay)
        except openai.APIStatusError as exc:
            raise LLMRequestError(
                f"Requête refusée par le fournisseur du LLM (HTTP {exc.status_code})."
            ) from exc
    raise AssertionError("inatteignable : la boucle renvoie ou lève toujours")
