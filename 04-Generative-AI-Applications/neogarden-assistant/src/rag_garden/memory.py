"""Historique de conversation : conversion vers le format de LangChain."""

from collections.abc import Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def to_langchain_messages(messages: Sequence[Mapping[str, str]]) -> list[BaseMessage]:
    """Convertit des messages `{"role", "content"}` en messages LangChain.

    « user » devient un `HumanMessage`, tout autre rôle un `AIMessage`.
    """
    return [
        HumanMessage(content=message["content"])
        if message["role"] == "user"
        else AIMessage(content=message["content"])
        for message in messages
    ]
