# Banking77 · Intent Router

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-2b5b9e?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.32+-ff4b4b?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-LinearSVC-f7931e?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLP-NLTK-lightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/status-live-brightgreen?style=flat-square"/>
</p>

<p align="center">
  <a href="https://banking77-intent-router-mgkdtpmzzvkqmjldzbnnpt.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Open%20App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"/>
  </a>
</p>

<p align="center">
  <b>Automated intent classification system for banking customer requests</b><br/>
  <sub>Built on the Banking77 dataset · 77 intents · TF-IDF + LinearSVC</sub>
</p>

---

## Overview

Banking77 Intent Router is a production-ready NLP application that automatically classifies customer banking requests into **77 fine-grained intent categories**. It routes each message to the appropriate department in real time, reducing manual triage and improving response efficiency.

The interface is designed with a **luxury banking aesthetic** — dark navy, gold accents, and refined typography — making it portfolio-ready and demo-friendly.

---

## Live Demo

🔗 **[banking77-intent-router.streamlit.app](https://banking77-intent-router-mgkdtpmzzvkqmjldzbnnpt.streamlit.app/)**

> Try it directly in your browser — no installation required.

---

## Features

- **77-class intent classification** trained on the Banking77 benchmark dataset
- **TF-IDF vectorization** with custom stopword filtering for banking vocabulary
- **Lemmatization pipeline** (NLTK WordNetLemmatizer)
- **Real-time prediction** with preprocessed token display
- **Luxury UI** built with Streamlit + custom CSS (Cormorant Garamond · Montserrat · navy/gold palette)
- **Quick scenario buttons** for instant demo

---

## Model

| Component     | Details                          |
|---------------|----------------------------------|
| Vectorizer    | TF-IDF (unigrams + bigrams)      |
| Classifier    | LinearSVC                        |
| Dataset       | Banking77 (13,083 samples)       |
| Intents       | 77 categories                    |
| Language      | English                          |

---

## Project Structure

```
banking77-intent-router/
├── app.py                    # Streamlit application
├── deploy.py                 # Deployment script (generates app.py + launches)
├── best_model.pkl            # Trained LinearSVC model
├── tfidf_vectorizer (1).pkl  # Fitted TF-IDF vectorizer
├── label_map.pkl             # Intent label mapping
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/akbrejnev200-rgb/banking77-intent-router.git
cd banking77-intent-router
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python deploy.py
```

Then open your browser at `http://localhost:8501`

---

## Demo

> Type any banking-related message and the model will classify the intent instantly.

**Example inputs:**
- *"I haven't received my card after two weeks"* → `card_arrival`
- *"Why was I charged twice for the same transaction?"* → `extra_charge_on_statement`
- *"I need to block my lost card immediately"* → `lost_or_stolen_card`

---

## Author

**akbrejnev200-rgb** · [github.com/akbrejnev200-rgb](https://github.com/akbrejnev200-rgb)

---

<p align="center">
  <sub>Banking77 dataset · Made with Streamlit · 2026</sub>
</p>
