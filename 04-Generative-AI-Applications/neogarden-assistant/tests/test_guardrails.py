"""Tests du raccourci « salutations » (seul garde-fou existant a ce stade)."""

import pytest

from rag_garden.guardrails import GREETING_REPLY, is_greeting


@pytest.mark.parametrize(
    "text", ["bonjour", "Bonjour", "salut", "  hey  ", "merci beaucoup", "CC"]
)
def test_is_greeting_recognizes_known_greetings(text):
    assert is_greeting(text) is True


@pytest.mark.parametrize(
    "text", ["bonjour, avez-vous des tondeuses ?", "", "au revoir tout le monde"]
)
def test_is_greeting_rejects_non_greetings(text):
    assert is_greeting(text) is False


def test_is_greeting_ignores_trailing_punctuation():
    assert is_greeting("bonjour!") is True
    assert is_greeting("salut.") is True


def test_is_greeting_ignores_space_before_punctuation():
    # Bug connu (typographie française : espace avant « ! ») : "Bonjour !" doit
    # aussi être reconnu comme une salutation, comme "Bonjour!" l'est déjà.
    assert is_greeting("Bonjour !") is True


def test_greeting_reply_mentions_the_shop_topics():
    assert "produits" in GREETING_REPLY.lower()
