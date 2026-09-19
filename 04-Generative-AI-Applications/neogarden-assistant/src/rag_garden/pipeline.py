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
from rag_garden.retriever import build_history_aware_retriever, build_retriever


@dataclass(frozen=True)
class Answer:
    """Réponse de l'assistant et passages qui la fondent (vide pour une salutation)."""

    text: str
    sources: list[Document]


def load_retriever() -> BaseRetriever:
    """Partie lourde (embeddings + index), indépendante de la clé API."""
    return build_retriever(load_or_build_vectorstore())


def build_rag_chain(retriever: BaseRetriever, api_key: str) -> Runnable:
    """Partie légère : branche le LLM sur le retriever (reformulation + réponse)."""
    llm = build_llm(api_key)
    history_aware_retriever = build_history_aware_retriever(llm, retriever)
    return create_retrieval_chain(history_aware_retriever, build_answer_chain(llm))


def answer_question(
    chain: Runnable, question: str, chat_history: Sequence[BaseMessage]
) -> Answer:
    """Répond à une question : salutation directe, sinon passage par le RAG."""
    if guardrails.is_greeting(question):
        return Answer(text=guardrails.GREETING_REPLY, sources=[])
    response = invoke_with_retry(
        chain, {"input": question, "chat_history": list(chat_history)}
    )
    return Answer(text=response["answer"], sources=response["context"])
