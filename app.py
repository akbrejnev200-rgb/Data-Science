import streamlit as st
import os
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Loaders pour lire les documents
from langchain_community.document_loaders import DataFrameLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LLM via API et assemblage des chaînes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURATION DE LA PAGE STREAMLIT ---
st.set_page_config(page_title="NeoGarden Assistant", page_icon="🌿", layout="centered")
AVATARS = {"user": "🧑‍🌾", "assistant": "🌿"}
# Salutations simples : réponse directe, sans passer par le RAG (pas de retrieval inutile, pas de sources hors sujet)
GREETINGS = {"bonjour", "salut", "hello", "coucou", "hey", "bonsoir", "merci", "merci beaucoup", "au revoir", "ça va", "ca va", "cc"}
st.title("🌿 Assistant NeoGarden")
st.markdown("Posez vos questions sur nos produits, vos commandes ou nos conditions de retour !")

# --- BARRE LATÉRALE ---
api_key_input = os.getenv("OPENROUTER_API_KEY")
with st.sidebar:
    st.header("⚙️ Configuration")
    if api_key_input:
        st.success("Clé API OpenRouter chargée depuis .env")
    else:
        st.error("Aucune clé API trouvée. Ajoute OPENROUTER_API_KEY dans le fichier .env")
        st.markdown("Obtenez votre clé sur [openrouter.ai/keys](https://openrouter.ai/keys)")
    st.markdown("---")
    st.markdown("Cet assistant utilise un système RAG couplé au modèle `nvidia/nemotron-3-super-120b-a12b` via OpenRouter.")


# --- FONCTION DE MISE EN CACHE DU RAG ---
# st.cache_resource permet de ne charger la base FAISS qu'une seule fois au démarrage
@st.cache_resource(show_spinner=False)
def init_rag_chain(api_key):
    # 1. Chargement des documents Word
    faq_loader = Docx2txtLoader("NeoGarden_FAQ.docx")
    cgu_loader = Docx2txtLoader("NeoGarden_CGU.docx")
    retours_loader = Docx2txtLoader("NeoGarden_Politique_Retour.docx")

    docs_text = faq_loader.load() + cgu_loader.load() + retours_loader.load()

    # 2. Chargement du catalogue produit
    try:
        df_catalogue = pd.read_csv("Catalogue_Jardin_Amazon-2 - Catalogue.csv")
        df_catalogue['rag_content'] = df_catalogue.apply(
            lambda row: (
                f"Produit: {str(row.get('Titre Amazon', ''))} | "
                f"Catégorie: {str(row.get('Catégorie', ''))} | "
                f"Prix: {str(row.get('Prix', ''))} | "
                f"Marque: {str(row.get('Marque', ''))} | "
                f"Description: {str(row.get('Description', ''))} | "
                f"Caractéristiques: {str(row.get('Caractéristiques techniques', ''))}"
            ), axis=1
        )
        df_loader = DataFrameLoader(df_catalogue, page_content_column="rag_content")
        docs_catalogue = df_loader.load()
    except FileNotFoundError:
        docs_catalogue = []

    all_documents = docs_text + docs_catalogue

    # 3. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(all_documents)

    # 4. Embeddings et Base FAISS
    embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/sentence-camembert-base")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 5. Configuration du LLM via OpenRouter (endpoint compatible OpenAI)
    # Modèle fixe (pas de routage auto) : "openrouter/free" peut piocher un modèle
    # non-conversationnel (ex: un classifieur de sécurité) et casser la démo.
    llm = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    system_prompt = (
        "Tu es un assistant virtuel expert pour NeoGarden, une boutique en ligne de jardinage. "
        "Ton rôle est d'aider les clients de manière polie, claire et concise.\n\n"
        "RÈGLES IMPORTANTES :\n"
        "1. Utilise UNIQUEMENT le contexte fourni ci-dessous pour répondre.\n"
        "2. Ne fais pas de suppositions et n'invente pas de prix ou de produits.\n"
        "3. Si l'information n'est pas dans le contexte, réponds honnêtement que tu ne sais pas "
        "4. Réponds toujours en français.\n\n"
        "Contexte :\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # Reformule la question de suivi en question autonome AVANT de chercher dans FAISS
    # (ex: "il coûte combien ?" -> "Combien coûte la tondeuse Torvald ?")
    contextualize_q_system_prompt = (
        "Étant donné l'historique de conversation et la dernière question de l'utilisateur, "
        "laquelle peut faire référence à des éléments cités plus haut, reformule cette question "
        "pour qu'elle soit compréhensible seule, sans l'historique. "
        "Ne réponds PAS à la question, reformule-la seulement si nécessaire ; "
        "sinon renvoie-la telle quelle."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def invoke_with_retry(chain, payload, max_retries=2, delay=2):
    """Réessaie l'appel LLM en cas d'erreur transitoire (ex: 502 fournisseur surchargé)."""
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return chain.invoke(payload)
        except Exception as e:
            last_exception = e
            print(f"[Tentative {attempt + 1}/{max_retries + 1} échouée] {e}")
            if attempt < max_retries:
                time.sleep(delay)
    raise last_exception


def display_sources(sources):
    """Affiche les chunks récupérés par FAISS sous la réponse, pour prouver l'ancrage documentaire."""
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)} extraits utilisés)"):
        for i, doc in enumerate(sources, start=1):
            source_name = os.path.basename(doc.metadata.get("source", "Catalogue produits (CSV)"))
            st.markdown(f"**{i}. {source_name}**")
            st.caption(doc.page_content)


# --- INITIALISATION DE L'ÉTAT ET DU RAG ---
if "messages" not in st.session_state:
    # Message de bienvenue par défaut
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Bonjour ! Je suis l'assistant NeoGarden. Comment puis-je vous aider pour vos achats de jardinage ou le suivi de vos commandes ?"}
    ]

# On s'assure qu'une clé API est renseignée avant de charger le RAG
if api_key_input:
    with st.spinner("Chargement de la base de connaissances NeoGarden..."):
        try:
            rag_chain = init_rag_chain(api_key_input)
            st.session_state.rag_ready = True
        except Exception as e:
            st.error(f"Erreur d'initialisation : {e}")
            st.session_state.rag_ready = False
else:
    st.warning("👈 Veuillez configurer la clé API OpenRouter (fichier .env) pour activer le chatbot.")
    st.session_state.rag_ready = False

# --- AFFICHAGE DE L'HISTORIQUE DU CHAT ---
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS.get(message["role"])):
        st.markdown(message["content"])
        display_sources(message.get("sources"))

# --- GESTION DES NOUVEAUX MESSAGES ---
if prompt := st.chat_input("Tapez votre message ici (ex: Livrez-vous en Belgique ?)..."):

    if not st.session_state.rag_ready:
        st.error("Veuillez d'abord configurer la clé API OpenRouter.")
    else:
        # 1. Afficher le message de l'utilisateur
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Générer et afficher la réponse de l'assistant
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            if prompt.strip().lower().rstrip("!.?") in GREETINGS:
                # Salutation simple : pas de RAG, réponse fixe, aucune source à afficher
                answer = "Bonjour ! Comment puis-je vous aider concernant nos produits, vos commandes ou nos conditions de retour ?"
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                with st.spinner("Je cherche..."):
                    try:
                        # On convertit l'historique Streamlit au format que LangChain comprend
                        chat_history = []
                        # On exclut le tout premier message de bienvenue (index 0) et la question actuelle (le dernier élément)
                        for msg in st.session_state.messages[1:-1]:
                            if msg["role"] == "user":
                                chat_history.append(HumanMessage(content=msg["content"]))
                            else:
                                chat_history.append(AIMessage(content=msg["content"]))

                        # On envoie la question ET l'historique au LLM (avec retry sur erreur transitoire)
                        response = invoke_with_retry(rag_chain, {
                            "input": prompt,
                            "chat_history": chat_history
                        })
                        answer = response['answer']
                        sources = response['context']
                        st.markdown(answer)
                        display_sources(sources)
                        # On sauvegarde la réponse ET ses sources dans l'historique
                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                    except Exception as e:
                        # Détail technique en console pour le debug, jamais montré au client
                        print(f"[Erreur RAG] {e}")
                        friendly_message = (
                            "Désolé, je rencontre un problème technique momentané. "
                            "Merci de réessayer votre question dans quelques instants."
                        )
                        st.error(friendly_message)
                        # On garde une trace dans l'historique pour que la conversation reste cohérente
                        st.session_state.messages.append({"role": "assistant", "content": friendly_message})

st.divider()
st.caption("🌿 NeoGarden est une boutique fictive — démo d'un système RAG (Retrieval-Augmented Generation).")