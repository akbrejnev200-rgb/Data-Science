"""Contrôles sur l'entrée et la sortie de l'assistant.

Trois garde-fous, mesurés sur le golden set avant d'être activés (Phase 5bis,
docs/PLAN.md) pour éviter de refuser de vraies questions par erreur :

1. Injection de prompt (entrée) : détection par motifs, avant toute recherche.
2. Question hors sujet (entrée) : absence de recouvrement avec le vocabulaire
   du corpus. Un seuil de similarité vectorielle a été essayé et abandonné :
   mesuré sur de vraies questions, il pénalisait autant une question de suivi
   légitime ("Et combien ça coûte ?") qu'une vraie question hors sujet.
3. Réponse non ancrée (sortie) : heuristique locale (recouvrement lexical
   avec le contexte retrouvé), pas un second appel LLM — pour ne pas doubler
   le coût de chaque question réelle posée dans l'application.

Plus ancien : le raccourci pour les salutations simples (réponse directe,
sans passer par le RAG).
"""

import re
from collections.abc import Iterable

from langchain_core.documents import Document

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
    """Vrai si le message est une simple salutation (casse/ponctuation ignorées)."""
    return text.strip().lower().rstrip(" !.?") in GREETINGS


# --- Garde-fou 1 : injection de prompt ---------------------------------------
# Détection par motifs : première ligne de défense, contournable par une
# formulation suffisamment différente. Motifs choisis à partir des tentatives
# du golden set (evaluation/golden_set.csv, questions 18-20) puis généralisés.
_INJECTION_PATTERNS = (
    re.compile(r"ignor(e|ez).{0,25}(instruction|consigne)", re.IGNORECASE),
    re.compile(r"ignore.{0,20}(previous|above).{0,20}instruction", re.IGNORECASE),
    re.compile(
        r"(répète|repete|révèle|revele|donne|montre)"
        r".{0,30}(instruction|prompt|consigne).{0,20}(système|systeme)",
        re.IGNORECASE,
    ),
    re.compile(r"system prompt", re.IGNORECASE),
    re.compile(r"tu (es|n'es plus).{0,25}(maintenant|DAN)", re.IGNORECASE),
    re.compile(r"sans (aucune )?restriction", re.IGNORECASE),
    re.compile(r"\bDAN\b"),
    re.compile(r"disregard.{0,20}(previous|above).{0,20}instruction", re.IGNORECASE),
    re.compile(r"pretend (you are|to be)", re.IGNORECASE),
)

INJECTION_REPLY = (
    "Je ne peux pas suivre cette instruction. Je reste l'assistant NeoGarden : "
    "posez-moi une question sur nos produits, la livraison, le paiement ou les "
    "retours."
)


def is_prompt_injection(text: str) -> bool:
    """Vrai si le message ressemble à une tentative de manipulation du prompt."""
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


# --- Garde-fou 2 : question hors sujet ---------------------------------------
_GENERIC_STOPWORDS = frozenset(
    {
        "le",
        "la",
        "les",
        "de",
        "des",
        "du",
        "et",
        "un",
        "une",
        "est",
        "dans",
        "pour",
        "ne",
        "pas",
        "que",
        "qui",
        "sur",
        "avec",
        "ce",
        "cette",
        "ces",
        "au",
        "aux",
        "en",
        "il",
        "elle",
        "ils",
        "elles",
        "je",
        "tu",
        "vous",
        "nous",
        "son",
        "sa",
        "ses",
        "leur",
        "leurs",
        "plus",
        "fait",
        "peut",
        "sont",
        "vos",
        "votre",
        "notre",
        "nos",
        "mais",
        "ou",
        "donc",
        "car",
        "tout",
        "tous",
        "toute",
        "toutes",
        "par",
        "si",
        "etre",
        "avoir",
        "a",
        "ete",
        "sans",
        "entre",
        "apres",
        "avant",
        "depuis",
        "jusqu",
        "ainsi",
        "comme",
        "cet",
        "meme",
        "autre",
        "bien",
        "quel",
        "quels",
        "quelle",
        "quelles",
        "comment",
        "combien",
        "pourquoi",
        "quoi",
        "avez",
        "suis",
        "puis",
    }
)

OFF_TOPIC_REPLY = (
    "Je ne trouve rien dans mes sources qui corresponde à votre question. "
    "Je peux vous aider sur nos produits, la livraison, le paiement ou les "
    "retours."
)


def _significant_words(text: str) -> set[str]:
    """Mots de plus de 3 lettres, hors mots outils génériques."""
    words = re.findall(r"[a-zàâäéèêëïîôöùûüç]+", text.lower())
    return {word for word in words if len(word) > 3 and word not in _GENERIC_STOPWORDS}


def build_domain_vocabulary(documents: Iterable[Document]) -> frozenset[str]:
    """Vocabulaire du corpus (FAQ, CGU, retours, catalogue), calculé une fois."""
    vocabulary: set[str] = set()
    for document in documents:
        vocabulary |= _significant_words(document.page_content)
    return frozenset(vocabulary)


def is_off_topic(question: str, vocabulary: frozenset[str]) -> bool:
    """Vrai si aucun mot significatif de la question n'appartient au corpus.

    Mesuré sur le golden set (docs/PLAN.md) : 0 faux positif sur les 14
    questions normales, 3/3 questions hors périmètre détectées. Limite connue :
    une question sans aucun mot de plus de 3 lettres propre au sujet (ex.
    "ça marche comment ?") est aussi classée hors sujet, même légitime.
    """
    return not (_significant_words(question) & vocabulary)


# --- Garde-fou 3 : réponse non ancrée dans le contexte -----------------------
# Heuristique locale (pas un second appel LLM) : recouvrement lexical entre la
# réponse et les passages retrouvés. Une vraie vérification par juge LLM serait
# plus fine mais doublerait le coût de chaque question réelle de l'application.
_REFUSAL_MARKERS = (
    "ne sais pas",
    "ne dispose pas",
    "n'ai pas cette information",
    "pas d'information",
    "hors de mon domaine",
    "hors sujet",
)

NOT_GROUNDED_REPLY = (
    "Je ne suis pas certain que ma réponse s'appuie bien sur mes sources "
    "actuelles. Pourriez-vous reformuler votre question, ou préciser le "
    "produit ou document concerné ?"
)


def is_grounded(answer: str, sources: list[Document], min_overlap: float = 0.3) -> bool:
    """Vrai si la réponse partage assez de mots avec le contexte retrouvé.

    Un refus ("je ne sais pas...") est toujours considéré ancré : il n'affirme
    rien qui devrait provenir du contexte.
    """
    lowered = answer.lower()
    if any(marker in lowered for marker in _REFUSAL_MARKERS):
        return True
    answer_words = _significant_words(answer)
    if not answer_words:
        return True
    context_words = _significant_words(
        " ".join(document.page_content for document in sources)
    )
    overlap = len(answer_words & context_words) / len(answer_words)
    return overlap >= min_overlap
