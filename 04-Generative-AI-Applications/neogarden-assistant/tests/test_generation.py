"""Tests de la politique de retry : erreurs simulees, aucun appel reseau.

Formalise le script de verification utilise pendant la Phase 2 (jamais garde
dans le depot) : c'est la logique la plus fragile du projet, elle merite une
couverture permanente plutot qu'une verification ponctuelle.
"""

import httpx
import openai
import pytest
from langchain_openai import ChatOpenAI

from rag_garden import generation
from rag_garden.errors import LLMRequestError, LLMUnavailableError

REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def status_error(error_cls: type, code: int) -> Exception:
    """Construit une vraie exception openai.* avec un code HTTP donne, sans reseau."""
    return error_cls(
        "erreur simulée", response=httpx.Response(code, request=REQUEST), body=None
    )


@pytest.fixture(scope="module")
def llm() -> ChatOpenAI:
    # Sert uniquement a sa methode de parsing d'erreur ; aucun appel reseau n'est fait.
    return ChatOpenAI(model="x", api_key="k", base_url="https://openrouter.ai/api/v1")


def embedded_error(llm: ChatOpenAI, body: dict) -> ValueError:
    """La vraie exception que LangChain leve pour une erreur dans un corps 200."""
    try:
        llm._create_chat_result(body)
    except ValueError as exc:
        return exc
    raise AssertionError("LangChain n'a pas levé d'erreur pour ce corps de réponse")


class FlakyChain:
    """Simule une chaine qui leve les erreurs donnees dans l'ordre, puis reussit."""

    def __init__(self, errors: list[Exception]):
        self.errors = list(errors)
        self.calls = 0

    def invoke(self, payload: dict) -> dict:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return {"answer": "ok", "context": []}


def test_succeeds_immediately_without_error():
    chain = FlakyChain([])

    result = generation.invoke_with_retry(chain, {}, max_retries=2, delay=0)

    assert result == {"answer": "ok", "context": []}
    assert chain.calls == 1


def test_retries_on_transient_sdk_error_then_succeeds():
    chain = FlakyChain([status_error(openai.InternalServerError, 502)])

    generation.invoke_with_retry(chain, {}, max_retries=2, delay=0)

    assert chain.calls == 2


def test_retries_on_network_error_then_rate_limit_then_succeeds():
    chain = FlakyChain(
        [
            openai.APIConnectionError(request=REQUEST),
            status_error(openai.RateLimitError, 429),
        ]
    )

    generation.invoke_with_retry(chain, {}, max_retries=2, delay=0)

    assert chain.calls == 3


def test_gives_up_after_max_retries_on_persistent_5xx():
    chain = FlakyChain([status_error(openai.InternalServerError, 502)] * 5)

    with pytest.raises(LLMUnavailableError):
        generation.invoke_with_retry(chain, {}, max_retries=2, delay=0)

    assert chain.calls == 3


def test_invalid_api_key_is_not_retried():
    chain = FlakyChain([status_error(openai.AuthenticationError, 401)])

    with pytest.raises(LLMRequestError):
        generation.invoke_with_retry(chain, {}, max_retries=2, delay=0)

    assert chain.calls == 1


def test_bad_request_is_not_retried():
    chain = FlakyChain([status_error(openai.BadRequestError, 400)])

    with pytest.raises(LLMRequestError):
        generation.invoke_with_retry(chain, {}, max_retries=2, delay=0)

    assert chain.calls == 1


def test_unexpected_bug_is_neither_retried_nor_hidden():
    chain = FlakyChain([ValueError("bug inattendu")])

    with pytest.raises(ValueError):
        generation.invoke_with_retry(chain, {}, max_retries=2, delay=0)

    assert chain.calls == 1


def test_error_embedded_in_200_body_is_retried(llm):
    body = {"error": {"message": "Provider returned error", "code": 502}}
    chain = FlakyChain([embedded_error(llm, body)])

    generation.invoke_with_retry(chain, {}, max_retries=2, delay=0)

    assert chain.calls == 2


def test_persistent_embedded_error_gives_up(llm):
    body = {"error": {"message": "Provider returned error", "code": 502}}
    chain = FlakyChain([embedded_error(llm, body) for _ in range(5)])

    with pytest.raises(LLMUnavailableError):
        generation.invoke_with_retry(chain, {}, max_retries=2, delay=0)

    assert chain.calls == 3


def test_embedded_credit_error_is_not_retried(llm):
    body = {"error": {"message": "Insufficient credits", "code": 402}}
    chain = FlakyChain([embedded_error(llm, body)])

    with pytest.raises(LLMRequestError):
        generation.invoke_with_retry(chain, {}, max_retries=2, delay=0)

    assert chain.calls == 1


def test_valueerror_without_code_is_not_retried():
    chain = FlakyChain([ValueError({"message": "?"})])

    with pytest.raises(LLMRequestError):
        generation.invoke_with_retry(chain, {}, max_retries=2, delay=0)

    assert chain.calls == 1
