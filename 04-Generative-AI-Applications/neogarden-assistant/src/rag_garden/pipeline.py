"""Orchestration du RAG complet : base de connaissances, chaîne, réponse."""

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_classic.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

from rag_garden import guardrails
from rag_garden.embeddings import load_or_build_vectorstore
from rag_garden.generation import build_answer_chain, build_llm, invoke_with_retry
from rag_garden.ingestion import load_documents
from rag_garden.retriever import build_history_aware_retriever, build_retriever


@dataclass(frozen=True)
class Answer:
    """Réponse de l'assistant et passages qui la fondent (vide pour un refus)."""

    text: str
    sources: list[Document]


def load_retriever() -> BaseRetriever:
    """Partie lourde (embeddings + index), indépendante de la clé API."""
    return build_retriever(load_or_build_vectorstore())


def load_domain_vocabulary() -> frozenset[str]:
    """Vocabulaire du corpus, pour le garde-fou "question hors sujet"."""
    return guardrails.build_domain_vocabulary(load_documents())


def build_rag_chain(retriever: BaseRetriever, api_key: str) -> Runnable:
    """Partie légère : branche le LLM sur le retriever (reformulation + réponse)."""
    llm = build_llm(api_key)
    history_aware_retriever = build_history_aware_retriever(llm, retriever)
    return create_retrieval_chain(history_aware_retriever, build_answer_chain(llm))


def answer_question(
    chain: Runnable,
    question: str,
    chat_history: Sequence[BaseMessage],
    domain_vocabulary: frozenset[str],
) -> Answer:
    """Répond à une question : garde-fous d'entrée, salutation, RAG, garde-fou sortie.

    Les garde-fous d'entrée (injection, hors sujet) évitent aussi un appel
    inutile au LLM : ils s'appliquent avant `invoke_with_retry`.
    """
    if guardrails.is_prompt_injection(question):
        return Answer(text=guardrails.INJECTION_REPLY, sources=[])
    if guardrails.is_greeting(question):
        return Answer(text=guardrails.GREETING_REPLY, sources=[])
    if guardrails.is_off_topic(question, domain_vocabulary):
        return Answer(text=guardrails.OFF_TOPIC_REPLY, sources=[])
    response = invoke_with_retry(
        chain, {"input": question, "chat_history": list(chat_history)}
    )
    answer_text, sources = response["answer"], response["context"]
    if not guardrails.is_grounded(answer_text, sources):
        return Answer(text=guardrails.NOT_GROUNDED_REPLY, sources=sources)
    return Answer(text=answer_text, sources=sources)
