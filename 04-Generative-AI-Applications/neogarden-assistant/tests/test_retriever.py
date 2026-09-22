"""Tests du retriever : integration FAISS + retriever, avec des embeddings factices.

Les embeddings sont simules (bases sur des mots-cles) pour ne telecharger ni
executer aucun modele reel : seule la logique de recherche est testee ici.
"""

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_garden.retriever import build_retriever

KEYWORDS = ["livraison", "retour", "paiement", "garantie"]


class FakeEmbeddings(Embeddings):
    """Mot-cle present -> 1.0 sur sa dimension, sinon 0.0. Deterministe, sans reseau."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        return [1.0 if keyword in lowered else 0.0 for keyword in KEYWORDS]


@pytest.fixture
def vectorstore() -> FAISS:
    documents = [
        Document(page_content="La livraison standard coûte 4,90 euros."),
        Document(page_content="Le retour est gratuit sous 30 jours."),
        Document(page_content="Le paiement se fait par carte ou virement."),
        Document(page_content="La garantie couvre deux ans."),
    ]
    return FAISS.from_documents(documents, FakeEmbeddings())


def test_build_retriever_returns_k_documents(vectorstore):
    retriever = build_retriever(vectorstore, k=2)

    results = retriever.invoke("Combien coûte la livraison ?")

    assert len(results) == 2


def test_build_retriever_returns_most_relevant_document_first(vectorstore):
    retriever = build_retriever(vectorstore, k=1)

    results = retriever.invoke("Quelles sont les conditions de garantie ?")

    assert "garantie" in results[0].page_content.lower()
