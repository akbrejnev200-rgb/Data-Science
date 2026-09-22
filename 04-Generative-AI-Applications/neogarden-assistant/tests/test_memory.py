"""Tests de la conversion de l'historique Streamlit vers les messages LangChain."""

from langchain_core.messages import AIMessage, HumanMessage

from rag_garden.memory import to_langchain_messages


def test_to_langchain_messages_converts_user_and_assistant_roles():
    messages = [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Bonjour ! Comment puis-je vous aider ?"},
    ]

    result = to_langchain_messages(messages)

    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "Bonjour"
    assert isinstance(result[1], AIMessage)
    assert result[1].content == "Bonjour ! Comment puis-je vous aider ?"


def test_to_langchain_messages_treats_any_non_user_role_as_assistant():
    messages = [{"role": "system", "content": "Message inattendu"}]

    result = to_langchain_messages(messages)

    assert isinstance(result[0], AIMessage)


def test_to_langchain_messages_handles_empty_history():
    assert to_langchain_messages([]) == []
