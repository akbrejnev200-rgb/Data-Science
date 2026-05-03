"""
deploy.py — Script de déploiement Banking77
Lancer depuis le terminal : python deploy.py
"""

import sys
import subprocess

APP_CODE = '''\
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt_tab", quiet=True)

st.set_page_config(
    page_title="Banking77 · Intent Router",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Montserrat:wght@300;400;500&display=swap');

:root {
    --navy:  #0b1628;
    --navy2: #112040;
    --navy3: #1a2f55;
    --gold:  #c9a84c;
    --gold2: #e8c97a;
    --gold3: #f5e6b8;
    --cream: #faf7f0;
    --muted: #8a9ab8;
    --border: rgba(201,168,76,0.25);
}

/* ── Reset global Streamlit ── */
html, body { background-color: #0b1628 !important; }

.stApp {
    background: linear-gradient(145deg, #0b1628 0%, #112040 50%, #0d1e3a 100%) !important;
    min-height: 100vh !important;
}

/* Écrase TOUS les textes Streamlit */
.stApp, .stApp p, .stApp span, .stApp div,
.stApp label, .stApp h1, .stApp h2, .stApp h3,
[class*="css"], .element-container {
    color: #faf7f0 !important;
    font-family: 'Montserrat', 'Georgia', sans-serif !important;
}

.block-container {
    padding-top: 56px !important;
    max-width: 720px !important;
    position: relative !important;
    z-index: 1 !important;
}

/* ── Header ── */
.lux-header {
    text-align: center !important;
    margin-bottom: 56px !important;
    padding-bottom: 32px !important;
}
.lux-eyebrow {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.6rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.35em !important;
    text-transform: uppercase !important;
    color: #c9a84c !important;
    margin-bottom: 14px !important;
    display: block !important;
}
.lux-title {
    font-family: 'Cormorant Garamond', 'Georgia', serif !important;
    font-size: 3.2rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.06em !important;
    color: #faf7f0 !important;
    margin: 0 0 6px 0 !important;
    line-height: 1.1 !important;
}
.lux-title em {
    font-style: italic !important;
    color: #e8c97a !important;
}
.lux-subtitle {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #8a9ab8 !important;
    margin-top: 10px !important;
}
.lux-divider {
    display: flex !important;
    align-items: center !important;
    gap: 14px !important;
    margin-top: 28px !important;
    justify-content: center !important;
}
.lux-divider-line {
    height: 1px !important;
    width: 80px !important;
    background: linear-gradient(90deg, transparent, #c9a84c, transparent) !important;
}
.lux-divider-gem {
    color: #c9a84c !important;
    font-size: 0.7rem !important;
}

/* ── Card ── */
.lux-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%) !important;
    border: 1px solid rgba(201,168,76,0.25) !important;
    border-radius: 2px !important;
    padding: 36px 40px !important;
    margin-bottom: 24px !important;
    position: relative !important;
    overflow: hidden !important;
}
.lux-card::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important; left: 0 !important; right: 0 !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, #c9a84c, transparent) !important;
}

/* ── Label ── */
.lux-label {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.58rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3em !important;
    text-transform: uppercase !important;
    color: #c9a84c !important;
    margin-bottom: 16px !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}
.lux-label::after {
    content: '' !important;
    flex: 1 !important;
    height: 1px !important;
    background: rgba(201,168,76,0.25) !important;
}

/* ── Textarea ── */
.stTextArea > div > div > textarea {
    background: rgba(11,22,40,0.8) !important;
    border: 1px solid rgba(201,168,76,0.25) !important;
    border-radius: 2px !important;
    color: #faf7f0 !important;
    font-family: 'Cormorant Garamond', 'Georgia', serif !important;
    font-size: 1.05rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.02em !important;
    caret-color: #c9a84c !important;
}
.stTextArea > div > div > textarea::placeholder {
    color: rgba(138,154,184,0.5) !important;
    font-style: italic !important;
}
.stTextArea > div > div > textarea:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 1px rgba(201,168,76,0.15) !important;
    outline: none !important;
}

/* ── Bouton principal ── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #c9a84c 0%, #e8c97a 100%) !important;
    color: #0b1628 !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.6rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.28em !important;
    text-transform: uppercase !important;
    padding: 14px 36px !important;
    box-shadow: 0 4px 20px rgba(201,168,76,0.3) !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 30px rgba(201,168,76,0.5) !important;
    opacity: 0.9 !important;
}

/* ── Boutons exemples ── */
div.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: #8a9ab8 !important;
    border: 1px solid rgba(138,154,184,0.2) !important;
    border-radius: 2px !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.55rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.12em !important;
    padding: 8px 10px !important;
    transition: all 0.25s ease !important;
}
div.stButton > button[kind="secondary"]:hover {
    border-color: #c9a84c !important;
    color: #e8c97a !important;
    background: rgba(201,168,76,0.05) !important;
}

/* ── Résultat ── */
.result-card {
    background: linear-gradient(135deg, #1a2f55 0%, #112040 100%) !important;
    border: 1px solid #c9a84c !important;
    border-radius: 2px !important;
    padding: 36px 40px !important;
    position: relative !important;
    overflow: hidden !important;
    margin-top: 28px !important;
}
.result-card::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important; left: 0 !important; right: 0 !important;
    height: 2px !important;
    background: linear-gradient(90deg, #c9a84c, #e8c97a, #c9a84c) !important;
}
.result-tag {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.55rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3em !important;
    text-transform: uppercase !important;
    color: #c9a84c !important;
    margin-bottom: 20px !important;
    display: block !important;
}
.result-intent {
    font-family: 'Cormorant Garamond', 'Georgia', serif !important;
    font-size: 2.4rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.04em !important;
    color: #faf7f0 !important;
    margin: 0 0 24px 0 !important;
    line-height: 1.1 !important;
}
.result-route {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.08em !important;
    color: #8a9ab8 !important;
    border-top: 1px solid rgba(201,168,76,0.15) !important;
    padding-top: 20px !important;
}
.result-route strong {
    color: #e8c97a !important;
    font-weight: 400 !important;
}
.preprocessed-lux {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.6rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.08em !important;
    color: rgba(138,154,184,0.7) !important;
    margin-top: 14px !important;
    padding: 12px 18px !important;
    border-left: 2px solid rgba(201,168,76,0.3) !important;
    background: rgba(255,255,255,0.01) !important;
    word-break: break-word !important;
}

/* ── Warning Streamlit restyled ── */
.stAlert {
    background: rgba(201,168,76,0.08) !important;
    border: 1px solid rgba(201,168,76,0.3) !important;
    border-radius: 2px !important;
    color: #e8c97a !important;
}

/* ── Footer ── */
.lux-footer {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.55rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: rgba(138,154,184,0.4) !important;
    margin-top: 64px !important;
    padding-top: 24px !important;
    border-top: 1px solid rgba(201,168,76,0.1) !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
}
.lux-footer-gem { color: #c9a84c !important; opacity: 0.4 !important; }

/* ── Masquer éléments natifs Streamlit ── */
#MainMenu, footer, header { visibility: hidden !important; }
div[data-testid="stDecoration"] { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer (1).pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
    return model, vectorizer, label_map


def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english")) - {
        "not", "no", "never", "without", "wrong", "failed"
    }
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


# ── Header
st.markdown("""
<div class="lux-header">
    <span class="lux-eyebrow">Private Banking Intelligence</span>
    <h1 class="lux-title">Banking<em>77</em></h1>
    <p class="lux-subtitle">Automated Intent Classification System</p>
    <div class="lux-divider">
        <div class="lux-divider-line"></div>
        <span class="lux-divider-gem">◈</span>
        <div class="lux-divider-line"></div>
    </div>
</div>
""", unsafe_allow_html=True)

try:
    model, vectorizer, label_map = load_assets()
except FileNotFoundError as e:
    st.error(f"Fichier manquant : {e}")
    st.stop()

if "msg" not in st.session_state:
    st.session_state.msg = ""

EXAMPLES = [
    "Card not received",
    "Charged twice",
    "Card declined",
    "Lost my card",
    "Change currency",
]
EXAMPLES_FULL = [
    "I haven't received my card after 2 weeks",
    "Why was I charged twice for the same payment?",
    "My card was declined but I have enough funds",
    "I lost my card and need to block it immediately",
    "Can I change my account currency?",
]

# ── Card saisie
st.markdown('<div class="lux-card">', unsafe_allow_html=True)
st.markdown('<span class="lux-label">Quick scenarios</span>', unsafe_allow_html=True)

cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    if cols[i].button(ex, key=f"ex_{i}"):
        st.session_state.msg = EXAMPLES_FULL[i]
        st.rerun()

st.markdown('<span class="lux-label" style="margin-top:28px;">Client message</span>', unsafe_allow_html=True)

user_input = st.text_area(
    label="client_message",
    value=st.session_state.msg,
    placeholder="Describe the client request in natural language...",
    height=130,
    label_visibility="collapsed",
)

st.markdown('</div>', unsafe_allow_html=True)

analyze = st.button("◈  Analyse intent", type="primary")

if analyze:
    if user_input.strip():
        cleaned    = preprocess(user_input)
        vectorized = vectorizer.transform([cleaned])
        pred_label  = model.predict(vectorized)[0]
        pred_intent = label_map.get(pred_label, "Unknown")
        intent_display = pred_intent.replace("_", " ").title()

        st.markdown(f"""
        <div class="result-card">
            <span class="result-tag">Detected intent</span>
            <p class="result-intent">{intent_display}</p>
            <div class="result-route">
                Ce message sera routé vers le service <strong>{intent_display}</strong>.
            </div>
        </div>
        <div class="preprocessed-lux">
            &#8594;&nbsp; {cleaned}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please enter a client message before analysing.")

model_name = type(model).__name__ if model else ""
st.markdown(f"""
<div class="lux-footer">
    <span>Banking77 &nbsp;·&nbsp; NLP Classification</span>
    <span class="lux-footer-gem">◈</span>
    <span>{model_name}</span>
</div>
""", unsafe_allow_html=True)
'''


def write_app():
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(APP_CODE)
    print("OK  app.py cree")


def launch():
    print("Lancement de Streamlit...\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless", "true"],
        check=True,
    )


if __name__ == "__main__":
    write_app()
    launch()
