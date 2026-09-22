"""Recherche des passages pertinents dans l'index vectoriel."""

from langchain_classic.chains import create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

from rag_garden import config
from rag_garden.prompts import build_contextualize_prompt


def build_retriever(vectorstore: FAISS, k: int = config.RETRIEVER_K) -> BaseRetriever:
    """Retriever qui renvoie les `k` passages les plus proches de la question."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_history_aware_retriever(
    llm: BaseChatModel, retriever: BaseRetriever
) -> Runnable:
    """Retriever qui reformule d'abord la question de suivi grâce à l'historique."""
    return create_history_aware_retriever(llm, retriever, build_contextualize_prompt())
