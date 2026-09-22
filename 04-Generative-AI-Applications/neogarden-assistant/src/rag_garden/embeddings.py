"""Embeddings et base vectorielle FAISS : construction, sauvegarde, rechargement.

L'index est sauvegardé sur disque avec une « empreinte » de ses entrées
(documents, taille de chunk, modèle d'embeddings). Au démarrage suivant, si
l'empreinte n'a pas changé, on recharge l'index au lieu de tout recalculer.
"""

import hashlib
import logging
from collections.abc import Iterable
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag_garden import config
from rag_garden.ingestion import load_documents, source_paths, split_documents

logger = logging.getLogger(__name__)

FINGERPRINT_FILE = "fingerprint.txt"


def get_embeddings(model_name: str = config.EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Charge le modèle d'embeddings (téléchargé une fois, puis en cache local)."""
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vectorstore(
    chunks: list[Document], embeddings: HuggingFaceEmbeddings
) -> FAISS:
    """Calcule les embeddings de tous les chunks et construit l'index FAISS."""
    return FAISS.from_documents(documents=chunks, embedding=embeddings)


def compute_fingerprint(
    files: Iterable[Path], chunk_size: int, chunk_overlap: int, embedding_model: str
) -> str:
    """Résume les entrées de l'index : si elles changent, l'index est périmé."""
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(f"{chunk_size}|{chunk_overlap}|{embedding_model}".encode())
    return digest.hexdigest()


def save_vectorstore(vectorstore: FAISS, index_dir: Path, fingerprint: str) -> None:
    """Sauvegarde l'index, puis l'empreinte EN DERNIER.

    L'empreinte sert de marqueur « sauvegarde complète » : si le programme
    s'arrête en cours d'écriture, elle manque et l'index sera reconstruit.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / FINGERPRINT_FILE).unlink(missing_ok=True)
    vectorstore.save_local(str(index_dir))
    (index_dir / FINGERPRINT_FILE).write_text(fingerprint, encoding="utf-8")


def load_vectorstore_if_fresh(
    index_dir: Path, embeddings: HuggingFaceEmbeddings, fingerprint: str
) -> FAISS | None:
    """Recharge l'index sauvegardé s'il est à jour, sinon renvoie None."""
    fingerprint_file = index_dir / FINGERPRINT_FILE
    if not fingerprint_file.exists():
        return None
    if fingerprint_file.read_text(encoding="utf-8").strip() != fingerprint:
        logger.info("Index périmé (documents ou paramètres modifiés).")
        return None
    # FAISS stocke ses métadonnées avec pickle : le charger exige ce drapeau.
    # C'est acceptable ici car l'index est produit par nous, en local, et n'est
    # jamais téléchargé depuis une source externe.
    return FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )


def load_or_build_vectorstore(
    data_dir: Path = config.DATA_DIR,
    index_dir: Path = config.VECTORSTORE_DIR,
    force_rebuild: bool = False,
) -> FAISS:
    """Renvoie l'index FAISS : rechargé s'il est à jour, sinon reconstruit."""
    embeddings = get_embeddings()
    fingerprint = compute_fingerprint(
        source_paths(data_dir),
        config.CHUNK_SIZE,
        config.CHUNK_OVERLAP,
        config.EMBEDDING_MODEL,
    )
    if not force_rebuild:
        vectorstore = load_vectorstore_if_fresh(index_dir, embeddings, fingerprint)
        if vectorstore is not None:
            logger.info("Index FAISS rechargé depuis %s", index_dir)
            return vectorstore

    chunks = split_documents(load_documents(data_dir))
    logger.info("Construction de l'index FAISS (%d chunks)...", len(chunks))
    vectorstore = build_vectorstore(chunks, embeddings)
    save_vectorstore(vectorstore, index_dir, fingerprint)
    return vectorstore
