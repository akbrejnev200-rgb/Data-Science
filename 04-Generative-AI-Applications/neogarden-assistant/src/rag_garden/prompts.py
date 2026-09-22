"""Prompts du RAG : consigne système et reformulation des questions de suivi."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# NB : les règles 3 et 4 sont collées sur une même ligne (il manque un saut de
# ligne après « ne sais pas »). Texte volontairement conservé tel quel : changer
# un prompt change le comportement du modèle, ça se corrige à part et se mesure
# avec l'évaluation (Phase 5).
SYSTEM_PROMPT = (
    "Tu es un assistant virtuel expert pour NeoGarden, une boutique en ligne de jardinage. "
    "Ton rôle est d'aider les clients de manière polie, claire et concise.\n\n"
    "RÈGLES IMPORTANTES :\n"
    "1. Utilise UNIQUEMENT le contexte fourni ci-dessous pour répondre.\n"
    "2. Ne fais pas de suppositions et n'invente pas de prix ou de produits.\n"
    "3. Si l'information n'est pas dans le contexte, réponds honnêtement que tu ne sais pas "
    "4. Réponds toujours en français.\n\n"
    "Contexte :\n{context}"
)

# Reformule une question de suivi en question autonome AVANT la recherche
# (ex : « il coûte combien ? » -> « Combien coûte la tondeuse Torvald ? »).
CONTEXTUALIZE_Q_SYSTEM_PROMPT = (
    "Étant donné l'historique de conversation et la dernière question de l'utilisateur, "
    "laquelle peut faire référence à des éléments cités plus haut, reformule cette question "
    "pour qu'elle soit compréhensible seule, sans l'historique. "
    "Ne réponds PAS à la question, reformule-la seulement si nécessaire ; "
    "sinon renvoie-la telle quelle."
)


def build_qa_prompt() -> ChatPromptTemplate:
    """Prompt de réponse : consigne système (avec le contexte), historique, question."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )


def build_contextualize_prompt() -> ChatPromptTemplate:
    """Prompt de reformulation : historique + dernière question."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
