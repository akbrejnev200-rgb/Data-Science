"""Erreurs métier du RAG.

Elles permettent de distinguer ce qu'on peut réessayer (fournisseur surchargé)
de ce qu'il est inutile de réessayer (clé API invalide), au lieu d'attraper
`Exception` partout.
"""


class RagError(Exception):
    """Erreur de base du projet."""


class KnowledgeBaseError(RagError):
    """Les documents ou l'index vectoriel n'ont pas pu être chargés."""


class LLMUnavailableError(RagError):
    """Le fournisseur du LLM est injoignable ou surchargé (erreur temporaire)."""


class LLMRequestError(RagError):
    """Le fournisseur du LLM a refusé la requête (clé invalide, requête mal formée).

    Réessayer ne servirait à rien.
    """
