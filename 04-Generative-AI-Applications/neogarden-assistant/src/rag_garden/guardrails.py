"""Contrôles sur l'entrée de l'utilisateur.

Implémenté aujourd'hui : le raccourci pour les salutations simples (réponse
directe, sans passer par le RAG : pas de recherche inutile, pas de sources hors
sujet).

NON implémenté (limite connue, voir README) : détection de prompt injection,
masquage de données personnelles, contrôle que la réponse est ancrée dans le
contexte. Les ajouter changerait le comportement (certaines questions seraient
refusées) : à décider à part.
"""

GREETINGS = frozenset(
    {
        "bonjour",
        "salut",
        "hello",
        "coucou",
        "hey",
        "bonsoir",
        "merci",
        "merci beaucoup",
        "au revoir",
        "ça va",
        "ca va",
        "cc",
    }
)

GREETING_REPLY = (
    "Bonjour ! Comment puis-je vous aider concernant nos produits, "
    "vos commandes ou nos conditions de retour ?"
)


def is_greeting(text: str) -> bool:
    """Vrai si le message est une simple salutation (casse et ponctuation finale ignorées)."""
    return text.strip().lower().rstrip("!.?") in GREETINGS
