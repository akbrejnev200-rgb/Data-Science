"""Tests des prompts : le contexte et la question doivent y apparaitre."""

from rag_garden.prompts import build_contextualize_prompt, build_qa_prompt


def test_qa_prompt_declares_context_history_and_question_variables():
    prompt = build_qa_prompt()

    assert "context" in prompt.input_variables
    assert "chat_history" in prompt.input_variables
    assert "input" in prompt.input_variables


def test_qa_prompt_formats_context_and_question_into_messages():
    prompt = build_qa_prompt()

    messages = prompt.format_messages(
        context="Le magasin est ouvert du lundi au samedi.",
        chat_history=[],
        input="Quels sont vos horaires ?",
    )

    system_message = messages[0].content
    human_message = messages[-1].content
    assert "Le magasin est ouvert du lundi au samedi." in system_message
    assert human_message == "Quels sont vos horaires ?"


def test_contextualize_prompt_declares_history_and_question_variables():
    prompt = build_contextualize_prompt()

    assert "chat_history" in prompt.input_variables
    assert "input" in prompt.input_variables
