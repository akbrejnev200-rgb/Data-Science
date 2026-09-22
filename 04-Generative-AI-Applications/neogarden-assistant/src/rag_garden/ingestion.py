"""Chargement et découpage des documents (FAQ, CGU, retours, catalogue produits)."""

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_garden import config
from rag_garden.errors import KnowledgeBaseError

logger = logging.getLogger(__name__)


def source_paths(data_dir: Path = config.DATA_DIR) -> list[Path]:
    """Liste les fichiers sources présents (documents Word + catalogue)."""
    names = (*config.DOCX_FILES, config.CATALOGUE_FILE)
    return [data_dir / name for name in names if (data_dir / name).exists()]


def load_docx_documents(data_dir: Path, filenames: Iterable[str]) -> list[Document]:
    """Charge les documents Word : un `Document` LangChain par fichier."""
    documents: list[Document] = []
    for filename in filenames:
        path = data_dir / filename
        if not path.exists():
            raise KnowledgeBaseError(f"Document introuvable : {path}")
        documents.extend(Docx2txtLoader(str(path)).load())
    return documents


def catalogue_row_to_text(row: pd.Series) -> str:
    """Transforme une ligne du catalogue en une phrase lisible par produit."""
    return (
        f"Produit: {str(row.get('Titre Amazon', ''))} | "
        f"Catégorie: {str(row.get('Catégorie', ''))} | "
        f"Prix: {str(row.get('Prix', ''))} | "
        f"Marque: {str(row.get('Marque', ''))} | "
        f"Description: {str(row.get('Description', ''))} | "
        f"Caractéristiques: {str(row.get('Caractéristiques techniques', ''))}"
    )


def load_catalogue_documents(csv_path: Path) -> list[Document]:
    """Charge le catalogue produits : une ligne du CSV devient un `Document`.

    Si le fichier est absent, on continue sans catalogue (comportement
    historique), mais en le signalant dans les logs au lieu de l'ignorer.
    """
    try:
        catalogue = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.warning(
            "Catalogue introuvable (%s) : index sans les produits.", csv_path
        )
        return []
    catalogue["rag_content"] = catalogue.apply(catalogue_row_to_text, axis=1)
    return DataFrameLoader(catalogue, page_content_column="rag_content").load()


def load_documents(data_dir: Path = config.DATA_DIR) -> list[Document]:
    """Charge tous les documents du RAG : textes Word, puis catalogue."""
    docx_documents = load_docx_documents(data_dir, config.DOCX_FILES)
    catalogue_documents = load_catalogue_documents(data_dir / config.CATALOGUE_FILE)
    return docx_documents + catalogue_documents


def split_documents(
    documents: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """Découpe les documents en chunks qui se chevauchent légèrement."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
