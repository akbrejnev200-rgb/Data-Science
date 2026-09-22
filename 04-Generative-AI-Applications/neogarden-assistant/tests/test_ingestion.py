"""Tests du chargement et du decoupage des documents (sans lire de vrai fichier)."""

import pandas as pd
from langchain_core.documents import Document

from rag_garden.ingestion import catalogue_row_to_text, split_documents


def test_split_documents_creates_multiple_chunks_within_size_limit():
    sentences = [f"Section {i} du document avec un peu de texte." for i in range(20)]
    documents = [Document(page_content=" ".join(sentences))]

    chunks = split_documents(documents, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 200 for chunk in chunks)


def test_split_documents_chunks_overlap():
    sentences = [f"Section {i} du document avec un peu de texte." for i in range(20)]
    documents = [Document(page_content=" ".join(sentences))]

    chunks = split_documents(documents, chunk_size=200, chunk_overlap=50)

    # Chaque paire de chunks voisins doit partager du texte (l'effet du chevauchement).
    # strict=False : chunks et chunks[1:] ont volontairement des tailles differentes.
    for first, second in zip(chunks, chunks[1:], strict=False):
        shared_words = set(first.page_content.split()) & set(
            second.page_content.split()
        )
        assert shared_words


def test_catalogue_row_to_text_formats_expected_fields():
    row = pd.Series(
        {
            "Titre Amazon": "Tondeuse Torvald",
            "Catégorie": "Jardin",
            "Prix": "199 €",
            "Marque": "Torvald",
            "Description": "Une tondeuse robuste.",
            "Caractéristiques techniques": "1200W",
        }
    )

    text = catalogue_row_to_text(row)

    assert text == (
        "Produit: Tondeuse Torvald | Catégorie: Jardin | Prix: 199 € | "
        "Marque: Torvald | Description: Une tondeuse robuste. | "
        "Caractéristiques: 1200W"
    )


def test_catalogue_row_to_text_handles_missing_fields():
    row = pd.Series({"Titre Amazon": "Produit Partiel"})

    text = catalogue_row_to_text(row)

    assert text.startswith("Produit: Produit Partiel |")
    assert "Catégorie: " in text
