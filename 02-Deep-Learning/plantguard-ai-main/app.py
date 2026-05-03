import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# MÉTRIQUES MODÈLE — À mettre à jour après entraînement
# ─────────────────────────────────────────
MODEL_ACCURACY  = 0.9894
MODEL_F1        = 0.99
MODEL_PRECISION = 0.99
MODEL_RECALL    = 0.99
NUM_CLASSES     = 29
NUM_IMAGES      = 67111

# ─────────────────────────────────────────
# CLASSES & CONSEILS
# ─────────────────────────────────────────
CLASS_INFO = {
    "Apple_Apple_Scab": {
        "name": "Apple — Apple Scab",
        "status": "diseased",
        "severity": "Modérée",
        "conseil": "Appliquer un fongicide à base de cuivre. Retirer les feuilles infectées et éviter l'humidité excessive.",
        "prevention": "Tailler les branches pour améliorer la circulation d'air."
    },
    "Apple_Black_Rot": {
        "name": "Apple — Black Rot",
        "status": "diseased",
        "severity": "Élevée",
        "conseil": "Retirer immédiatement les fruits et feuilles infectés. Traiter avec un fongicide homologué.",
        "prevention": "Maintenir une bonne hygiène du verger et éviter les blessures sur les fruits."
    },
    "Apple_Cedar_Apple_Rust": {
        "name": "Apple — Cedar Rust",
        "status": "diseased",
        "severity": "Modérée",
        "conseil": "Utiliser des fongicides préventifs au printemps. Supprimer les genévriers à proximité.",
        "prevention": "Planter des variétés résistantes."
    },
    "Apple_Healthy": {
        "name": "Apple — Healthy",
        "status": "healthy",
        "severity": "Aucune",
        "conseil": "Votre plante est en parfaite santé. Continuez les bonnes pratiques culturales.",
        "prevention": "Maintenez un arrosage régulier et une fertilisation adaptée."
    },
    "Bell_Pepper_Bacterial_Spot": {
        "name": "Bell Pepper — Bacterial Spot",
        "status": "diseased",
        "severity": "Modérée",
        "conseil": "Supprimer les feuilles infectées. Appliquer un traitement au cuivre.",
        "prevention": "Éviter les arrosages par aspersion et assurer une bonne rotation des cultures."
    },
    "Bell_Pepper_Healthy": {
        "name": "Bell Pepper — Healthy",
        "status": "healthy",
        "severity": "Aucune",
        "conseil": "Plante en bonne santé. Poursuivez les pratiques actuelles.",
        "prevention": "Surveillez régulièrement l'apparition de taches ou décolorations."
    },
    "Cherry_Healthy": {
        "name": "Cherry — Healthy",
        "status": "healthy",
        "severity": "Aucune",
        "conseil": "Plante saine. Continuez le suivi régulier.",
        "prevention": "Assurez une bonne aération et un sol bien drainé."
    },
    "Cherry_Powdery_Mildew": {
        "name": "Cherry — Powdery Mildew",
        "status": "diseased",
        "severity": "Modérée",
        "conseil": "Traiter avec du soufre ou un fongicide systémique. Améliorer la ventilation.",
        "prevention": "Éviter les excès d'azote et tailler régulièrement."
    },
    "Corn_Cercospora_Leaf_Spot": {
        "name": "Corn — Cercospora Leaf Spot",
        "status": "diseased",
        "severity": "Modérée",
        "conseil": "Appliquer un fongicide foliaire. Assurer une rotation des cultures.",
        "prevention": "Utiliser des semences résistantes et maintenir un espacement adéquat."
    },
    "Corn_Common_Rust": {
        "name": "Corn — Common Rust",
        "status": "diseased",
        "severity": "Modérée",
        "conseil": "Traiter avec des fongicides à base de triazole. Surveiller l'évolution.",
        "prevention": "Planter des variétés résistantes à la rouille."
    },
    "Corn_Healthy": {
        "name": "Corn — Healthy",
        "status": "healthy",
        "severity": "Aucune",
        "conseil": "Culture en bonne santé. Continuez la surveillance.",
        "prevention": "Maintenez une fertilisation équilibrée en azote."
    },
    "Corn_Northern_Leaf_Blight": {
        "name": "Corn — Northern Leaf Blight",
        "status": "diseased",
        "severity": "Élevée",
        "conseil": "Appliquer un fongicide dès les premiers symptômes. Réduire la densité de plantation.",
        "prevention": "Rotation des cultures et utilisation de variétés tolérantes."
    },
    "Grape_Black_Rot": {
        "name": "Grape — Black Rot",
        "status": "diseased",
        "severity": "Élevée",
        "conseil": "Retirer les grappes infectées. Traiter avec un fongicide à base de cuivre ou manèbe.",
        "prevention": "Tailler correctement pour favoriser l'aération et la pénétration de la lumière."
    },
    "Grape_Esca": {
        "name": "Grape — Esca (Black Measles)",
        "status": "diseased",
        "severity": "Élevée",
        "conseil": "Pas de traitement curatif connu. Retirer les plants sévèrement atteints.",
        "prevention": "Protéger les plaies de taille avec un mastic fongicide."
    },
    "Grape_Healthy": {
        "name": "Grape — Healthy",
        "status": "healthy",
        "severity": "Aucune",
        "conseil": "Vigne en parfaite santé. Continuez les pratiques actuelles.",
        "prevention": "Surveillance régulière et taille annuelle adaptée."
    },
    "Grape_Leaf_Blight": {
        "name": "Grape — Leaf Blight",
        "status": "diseased",
        "severity": "Modérée",
        "conseil": "Traiter avec des fongicides homologués. Supprimer les feuilles atteintes.",
        "prevention": "Éviter l'excès d'humidité foliaire."
    },
    "Orange_Haunglongbing": {
        "name": "Orange — Huanglongbing (HLB)",
        "status": "diseased",
        "severity": "Critique",
        "conseil": "Maladie incurable. Arracher et détruire les plants infectés immédiatement.",
        "prevention": "Contrôler le psylle asiatique des agrumes, vecteur de la maladie."
    },
    "Peach_Bacterial_Spot": {
        "name": "Peach — Bacterial Spot",
        "status": "diseased",
        "severity": "Modérée",
        "conseil": "Appliquer un traitement cuprique en période de repos végétatif.",
        "prevention": "Choisir des variétés résistantes et éviter les blessures mécaniques."
    },
    "Peach_Healthy": {
        "name": "Peach — Healthy",
        "status": "healthy",
        "severity": "Aucune",
        "conseil": "Arbre en bonne santé. Maintenez les soins actuels.",
        "prevention": "Surveillance mensuelle des feuilles et des fruits."
    },
    "Potato_Early_Blight": {
        "name": "Potato — Early Blight",
        "status": "diseased",
        "severity": "Modérée",
        "conseil": "Traiter avec un fongicide à base de chlorothalonil ou mancozèbe.",
        "prevention": "Rotation des cultures sur 3 ans minimum."
    },
    "Potato_Healthy": {
        "name": "Potato — Healthy",
        "status": "healthy",
        "severity": "Aucune",
        "conseil": "Culture saine. Continuez le suivi.",
        "prevention": "Maintenir un bon drainage et éviter l'excès d'azote."
    },
    "Potato_Late_Blight": {
        "name": "Potato — Late Blight",
        "status": "diseased",
        "severity": "Critique",
        "conseil": "Traitement fongicide immédiat. Détruire les plants fortement infectés.",
        "prevention": "Utiliser des plants certifiés et des variétés résistantes."
    },
    "Raspberry_Healthy": {
        "name": "Raspberry — Healthy",
        "status": "healthy",
        "severity": "Aucune",
        "conseil": "Plante saine. Poursuivez les soins habituels.",
        "prevention": "Tailler après récolte pour renouveler les cannes."
    },
    "Soybean_Healthy": {
        "name": "Soybean — Healthy",
        "status": "healthy",
        "severity": "Aucune",
        "conseil": "Culture en bonne santé. Continuez la surveillance.",
        "prevention": "Rotation des cultures et inoculation rhizobienne recommandées."
    },
    "Squash_Powdery_Mildew": {
        "name": "Squash — Powdery Mildew",
        "status": "diseased",
        "severity": "Modérée",
        "conseil": "Appliquer du soufre en poudre ou un fongicide systémique.",
        "prevention": "Espacer les plants pour une meilleure circulation d'air."
    },
    "Strawberry_Healthy": {
        "name": "Strawberry — Healthy",
        "status": "healthy",
        "severity": "Aucune",
        "conseil": "Plants en parfaite santé. Maintenez les pratiques actuelles.",
        "prevention": "Renouveler les plants tous les 3-4 ans."
    },
    "Strawberry_Leaf_Scorch": {
        "name": "Strawberry — Leaf Scorch",
        "status": "diseased",
        "severity": "Modérée",
        "conseil": "Retirer les feuilles atteintes. Traiter avec un fongicide adapté.",
        "prevention": "Éviter l'arrosage foliaire et assurer un bon drainage."
    },
    "Tomato_Healthy": {
        "name": "Tomato — Healthy",
        "status": "healthy",
        "severity": "Aucune",
        "conseil": "Plante en excellente santé. Continuez les soins.",
        "prevention": "Tuteurer correctement et surveiller les premiers signes de maladie."
    },
    "Tomato_Late_Blight": {
        "name": "Tomato — Late Blight",
        "status": "diseased",
        "severity": "Critique",
        "conseil": "Traitement fongicide immédiat. Retirer et détruire les parties atteintes.",
        "prevention": "Éviter l'humidité foliaire et assurer une bonne aération."
    },
}

# ─────────────────────────────────────────
# CSS GLOBAL
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

* { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0F1A0F;
    border-right: 1px solid #1E3A1E;
}
[data-testid="stSidebar"] * { color: #E8F5E8 !important; }

/* Main background */
.stApp { background: #F7F9F4; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }

/* Header */
.app-header {
    background: linear-gradient(135deg, #0F1A0F 0%, #1B3A1B 50%, #2D6A4F 100%);
    padding: 2.5rem 3rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(45,106,79,0.3) 0%, transparent 70%);
    border-radius: 50%;
}
.app-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #FFFFFF;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-subtitle {
    color: #95C99B;
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* KPI Cards */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E8EFE8;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-value {
    font-size: 2.2rem;
    font-weight: 600;
    color: #2D6A4F;
    line-height: 1;
}
.kpi-label {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.5rem;
}

/* Upload zone */
.upload-section {
    background: #FFFFFF;
    border: 2px dashed #C8DEC8;
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    transition: border-color 0.3s;
}
.upload-section:hover { border-color: #2D6A4F; }

/* Result cards */
.result-healthy {
    background: linear-gradient(135deg, #F0FAF4 0%, #E8F5EC 100%);
    border: 1px solid #95D5B2;
    border-left: 5px solid #2D6A4F;
    border-radius: 12px;
    padding: 2rem;
}
.result-diseased {
    background: linear-gradient(135deg, #FFF8F0 0%, #FFF0E8 100%);
    border: 1px solid #FFB347;
    border-left: 5px solid #E07B39;
    border-radius: 12px;
    padding: 2rem;
}
.result-critical {
    background: linear-gradient(135deg, #FFF0F0 0%, #FFE8E8 100%);
    border: 1px solid #FF9999;
    border-left: 5px solid #CC3333;
    border-radius: 12px;
    padding: 2rem;
}
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #1A2E1A;
    margin-bottom: 0.5rem;
}
.severity-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
}
.badge-healthy { background: #D4EDDA; color: #155724; }
.badge-moderate { background: #FFF3CD; color: #856404; }
.badge-high { background: #FFE0CC; color: #8B3A00; }
.badge-critical { background: #F8D7DA; color: #721C24; }

/* Section titles */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #1A2E1A;
    border-bottom: 2px solid #2D6A4F;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

/* Info box */
.info-box {
    background: #F0F7F0;
    border: 1px solid #C8DEC8;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}

/* Metric row */
.metric-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid #F0F0F0;
    font-size: 0.9rem;
}
.metric-label { color: #666; }
.metric-value { font-weight: 600; color: #2D6A4F; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.4rem; color: #95C99B; margin-bottom: 0.3rem;'>🌿 PlantGuard AI</div>
        <div style='font-size: 0.75rem; color: #6B8F6B; letter-spacing: 1px; text-transform: uppercase;'>Diagnostic System v3.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Diagnostic", "Performance du Modèle", "À propos"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.8rem; color: #6B8F6B; line-height: 1.8;'>
        <div style='margin-bottom: 0.5rem;'><strong style='color: #95C99B;'>Modèle</strong></div>
        EfficientNetB0<br>
        Transfer Learning<br>
        Fine-Tuning 2 phases<br><br>
        <div style='margin-bottom: 0.5rem;'><strong style='color: #95C99B;'>Dataset</strong></div>
        PlantVillage (Updated)<br>
        67 111 images<br>
        29 classes<br><br>
        <div style='margin-bottom: 0.5rem;'><strong style='color: #95C99B;'>Auteur</strong></div>
        Brejnev AKOUMANI<br>
        <a href='https://linkedin.com/in/akbrejnev' style='color: #95C99B;'>LinkedIn</a> · 
        <a href='https://github.com/akbrejnev200-rgb' style='color: #95C99B;'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.75rem; color: #6B8F6B; line-height: 1.8;'>
        <div style='margin-bottom: 0.5rem;'><strong style='color: #E8A87C;'>⚠️ Espèces supportées</strong></div>
        <div style='color: #95C99B;'>
        🍎 Pomme · 🌶️ Poivron<br>
        🍒 Cerise · 🌽 Maïs<br>
        🍇 Raisin · 🍑 Pêche<br>
        🥔 Pomme de terre<br>
        🍓 Fraise · 🍅 Tomate
        </div>
        <div style='margin-top:0.5rem; color:#888; font-size:0.7rem;'>
        Toute autre plante (manguier,<br>
        bananier, etc.) ne sera pas<br>
        reconnue correctement.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model_once():
    try:
        model_path = Path(__file__).parent / "saved_model"
        return tf.saved_model.load(str(model_path))
    except Exception as e:
        st.error(f"Erreur chargement modèle : {e}")
        return None

model = load_model_once()

# ─────────────────────────────────────────
# PAGE 1 — DIAGNOSTIC
# ─────────────────────────────────────────
if page == "Diagnostic":

    st.markdown("""
    <div class='app-header'>
        <div class='app-title'>PlantGuard AI</div>
        <div class='app-subtitle'>Système de diagnostic des maladies végétales par intelligence artificielle</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-value'>{MODEL_ACCURACY:.0%}</div>
            <div class='kpi-label'>Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-value'>{MODEL_F1:.0%}</div>
            <div class='kpi-label'>F1-Score</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-value'>{NUM_CLASSES}</div>
            <div class='kpi-label'>Classes</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-value'>{NUM_IMAGES:,}</div>
            <div class='kpi-label'>Images entraînées</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Avertissement espèces supportées
    st.markdown("""
    <div style='background:#1B3A1B; border:1px solid #2D6A4F; border-left:4px solid #E8A87C;
                border-radius:10px; padding:1rem 1.5rem; margin-bottom:1.5rem;'>
        <div style='color:#E8A87C; font-weight:600; font-size:0.85rem; margin-bottom:0.5rem;'>
            ⚠️ Espèces reconnues par ce modèle
        </div>
        <div style='color:#95C99B; font-size:0.82rem; line-height:1.8;'>
             Pomme &nbsp;·&nbsp;  Poivron &nbsp;·&nbsp;  Cerise &nbsp;·&nbsp; 
             Maïs &nbsp;·&nbsp;  Raisin &nbsp;·&nbsp;  Pêche &nbsp;·&nbsp; 
             Pomme de terre &nbsp;·&nbsp;  Fraise &nbsp;·&nbsp;  Tomate
        </div>
        <div style='color:#888; font-size:0.75rem; margin-top:0.5rem;'>
            Toute autre espèce (manguier, bananier, oranger…) ne sera pas reconnue et produira un résultat non fiable.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Upload section
    col_upload, col_result = st.columns([1, 1.2], gap="large")

    with col_upload:
        st.markdown("<div class='section-title'>Analyse d'image</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Téléversez une image de feuille",
            type=["jpg", "jpeg", "png"],
            help="Formats acceptés : JPG, JPEG, PNG"
        )

        if uploaded_file:
            img_display = Image.open(uploaded_file)
            st.image(img_display, caption="Image analysée", use_column_width=True)

    with col_result:
        st.markdown("<div class='section-title'>Résultat du diagnostic</div>", unsafe_allow_html=True)

        if uploaded_file and model:
            with st.spinner("Analyse en cours..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.008)
                    progress.progress(i + 1)

                img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
                img_array = np.array(img, dtype=np.float32)
                img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
                img_tensor = tf.expand_dims(img_array, axis=0)

                infer = model.signatures["serving_default"]
                output = infer(tf.constant(img_tensor))
                output_key = list(output.keys())[0]
                prediction = output[output_key].numpy()

                top3_idx = np.argsort(prediction[0])[::-1][:3]

                # Mapping index → clé CLASS_INFO (ordre alphabétique = ordre du training)
                TRAINING_CLASSES = [
                    "Apple_Apple Scab", "Apple_Black Rot", "Apple_Cedar Apple Rust", "Apple_Healthy",
                    "Bell Pepper_Bacterial Spot", "Bell Pepper_Healthy",
                    "Cherry_Healthy", "Cherry_Powdery Mildew",
                    "Corn (Maize)_Cercospora Leaf Spot", "Corn (Maize)_Common Rust",
                    "Corn (Maize)_Healthy", "Corn (Maize)_Northern Leaf Blight",
                    "Grape_Black Rot", "Grape_Esca (Black Measles)", "Grape_Healthy", "Grape_Leaf Blight",
                    "Peach_Bacterial Spot", "Peach_Healthy",
                    "Potato_Early Blight", "Potato_Healthy", "Potato_Late Blight",
                    "Strawberry_Healthy", "Strawberry_Leaf Scorch",
                    "Tomato_Bacterial Spot", "Tomato_Early Blight", "Tomato_Healthy",
                    "Tomato_Late Blight", "Tomato_Septoria Leaf Spot", "Tomato_Yellow Leaf Curl Virus"
                ]
                predicted_key = TRAINING_CLASSES[top3_idx[0]] if top3_idx[0] < len(TRAINING_CLASSES) else TRAINING_CLASSES[0]
                # Table de correspondance nom réel → clé CLASS_INFO
                KEY_MAP = {
                    "Apple_Apple Scab": "Apple_Apple_Scab",
                    "Apple_Black Rot": "Apple_Black_Rot",
                    "Apple_Cedar Apple Rust": "Apple_Cedar_Apple_Rust",
                    "Apple_Healthy": "Apple_Healthy",
                    "Bell Pepper_Bacterial Spot": "Bell_Pepper_Bacterial_Spot",
                    "Bell Pepper_Healthy": "Bell_Pepper_Healthy",
                    "Cherry_Healthy": "Cherry_Healthy",
                    "Cherry_Powdery Mildew": "Cherry_Powdery_Mildew",
                    "Corn (Maize)_Cercospora Leaf Spot": "Corn_Cercospora_Leaf_Spot",
                    "Corn (Maize)_Common Rust": "Corn_Common_Rust",
                    "Corn (Maize)_Healthy": "Corn_Healthy",
                    "Corn (Maize)_Northern Leaf Blight": "Corn_Northern_Leaf_Blight",
                    "Grape_Black Rot": "Grape_Black_Rot",
                    "Grape_Esca (Black Measles)": "Grape_Esca",
                    "Grape_Healthy": "Grape_Healthy",
                    "Grape_Leaf Blight": "Grape_Leaf_Blight",
                    "Peach_Bacterial Spot": "Peach_Bacterial_Spot",
                    "Peach_Healthy": "Peach_Healthy",
                    "Potato_Early Blight": "Potato_Early_Blight",
                    "Potato_Healthy": "Potato_Healthy",
                    "Potato_Late Blight": "Potato_Late_Blight",
                    "Strawberry_Healthy": "Strawberry_Healthy",
                    "Strawberry_Leaf Scorch": "Strawberry_Leaf_Scorch",
                    "Tomato_Bacterial Spot": "Tomato_Bacterial_Spot",
                    "Tomato_Early Blight": "Tomato_Early_Blight",
                    "Tomato_Healthy": "Tomato_Healthy",
                    "Tomato_Late Blight": "Tomato_Late_Blight",
                    "Tomato_Septoria Leaf Spot": "Tomato_Septoria_Leaf_Spot",
                    "Tomato_Yellow Leaf Curl Virus": "Tomato_Yellow_Leaf_Curl_Virus",
                }
                info_key = KEY_MAP.get(predicted_key, predicted_key)
                info = CLASS_INFO.get(info_key, list(CLASS_INFO.values())[0])
                confidence = prediction[0][top3_idx[0]] * 100

            # ── Seuils de confiance ──────────────────────
            THRESHOLD_HIGH   = 80   # diagnostic fiable
            THRESHOLD_MEDIUM = 50   # diagnostic probable

            # Severity badge
            severity = info.get("severity", "Modérée")
            if info["status"] == "healthy":
                result_class = "result-healthy"
                badge_class = "badge-healthy"
            elif severity == "Critique":
                result_class = "result-critical"
                badge_class = "badge-critical"
            elif severity == "Élevée":
                result_class = "result-diseased"
                badge_class = "badge-high"
            else:
                result_class = "result-diseased"
                badge_class = "badge-moderate"

            if confidence >= THRESHOLD_HIGH:
                # ✅ Diagnostic fiable
                st.markdown(f"""
                <div class='{result_class}'>
                    <div class='result-title'>{info['name']}</div>
                    <span class='severity-badge {badge_class}'>Sévérité : {severity}</span>
                    <div style='font-size: 0.9rem; color: #444; margin-bottom: 1rem;'>
                        <strong>Recommandation :</strong><br>{info['conseil']}
                    </div>
                    <div style='font-size: 0.85rem; color: #666;'>
                        <strong>Prévention :</strong><br>{info['prevention']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif confidence >= THRESHOLD_MEDIUM:
                # ⚠️ Diagnostic probable
                st.markdown(f"""
                <div class='{result_class}' style='opacity: 0.9;'>
                    <div class='result-title'>{info['name']}</div>
                    <span class='severity-badge badge-moderate'>⚠️ Diagnostic probable</span>
                    <div style='background:#FFF8E1; border-left:3px solid #F59E0B; padding:0.7rem 1rem; border-radius:6px; font-size:0.85rem; color:#92400E; margin-bottom:1rem;'>
                        Confiance modérée ({confidence:.0f}%). Le diagnostic est possible mais non certain.
                        Veuillez photographier la feuille en gros plan sur fond neutre pour un meilleur résultat.
                    </div>
                    <div style='font-size: 0.9rem; color: #444; margin-bottom: 1rem;'>
                        <strong>Recommandation possible :</strong><br>{info['conseil']}
                    </div>
                    <div style='font-size: 0.85rem; color: #666;'>
                        <strong>Prévention :</strong><br>{info['prevention']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                # ❌ Image non reconnue
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#F8F9FA,#F0F0F0); border:1px solid #CCC;
                            border-left:5px solid #888; border-radius:12px; padding:2rem;'>
                    <div class='result-title' style='color:#555;'>🔍 Image non reconnue</div>
                    <div style='font-size:0.9rem; color:#666; margin-top:1rem; line-height:1.8;'>
                        Le modèle n'est pas suffisamment confiant pour établir un diagnostic 
                        (<strong>{confidence:.0f}%</strong> de confiance, seuil minimum : {THRESHOLD_MEDIUM}%).<br><br>
                        <strong>Pour obtenir un meilleur résultat :</strong><br>
                        • Photographiez <strong>une seule feuille</strong>, en gros plan<br>
                        • Utilisez un <strong>fond neutre</strong> (blanc ou sol uni)<br>
                        • Assurez un bon <strong>éclairage naturel</strong> sans ombre<br>
                        • Évitez les reflets et le flou<br><br>
                        <em>Ce système est entraîné sur des images en conditions contrôlées.</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Confidence gauge
            st.markdown("<br>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                number={'suffix': '%', 'font': {'size': 28, 'color': '#2D6A4F'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#888'},
                    'bar': {'color': '#2D6A4F'},
                    'bgcolor': '#F0F7F0',
                    'steps': [
                        {'range': [0, 50], 'color': '#FFE8E8'},
                        {'range': [50, 75], 'color': '#FFF3CD'},
                        {'range': [75, 100], 'color': '#D4EDDA'}
                    ],
                    'threshold': {
                        'line': {'color': '#1A2E1A', 'width': 2},
                        'thickness': 0.75,
                        'value': confidence
                    }
                },
                title={'text': "Indice de confiance", 'font': {'size': 14, 'color': '#666'}}
            ))
            fig_gauge.update_layout(
                height=200,
                margin=dict(t=40, b=0, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'family': 'DM Sans'}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Top 3
            st.markdown("<div style='font-size: 0.85rem; color: #888; margin-bottom: 0.5rem;'>Top 3 prédictions</div>", unsafe_allow_html=True)
            for i, idx in enumerate(top3_idx):
                if idx < len(TRAINING_CLASSES):
                    cls_key = TRAINING_CLASSES[idx]
                    cls_name = CLASS_INFO.get(cls_key, {}).get("name", cls_key)
                    prob = prediction[0][idx] * 100
                    st.progress(int(prob))
                    st.caption(f"{cls_name} — {prob:.1f}%")

        elif uploaded_file and not model:
            st.warning("Modèle non chargé. Vérifiez que `model.tflite` est dans le répertoire.")
        else:
            st.markdown("""
            <div class='info-box'>
                <div style='color: #2D6A4F; font-weight: 500; margin-bottom: 0.5rem;'>Comment utiliser PlantGuard AI</div>
                <div style='font-size: 0.85rem; color: #555; line-height: 1.7;'>
                    1. Photographiez une feuille de plante<br>
                    2. Téléversez l'image à gauche<br>
                    3. Obtenez un diagnostic instantané<br>
                    4. Suivez les recommandations agronomiques
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# PAGE 2 — PERFORMANCE DU MODÈLE
# ─────────────────────────────────────────
elif page == "Performance du Modèle":

    st.markdown("""
    <div class='app-header'>
        <div class='app-title'>Performance du Modèle</div>
        <div class='app-subtitle'>Métriques d'évaluation — EfficientNetB0 · Test Set · 29 classes</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Accuracy", f"{MODEL_ACCURACY:.1%}"),
        ("F1-Score", f"{MODEL_F1:.1%}"),
        ("Precision", f"{MODEL_PRECISION:.1%}"),
        ("Recall", f"{MODEL_RECALL:.1%}"),
    ]
    for col, (label, value) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-value'>{value}</div>
                <div class='kpi-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Encadré contexte honnête
    st.markdown("""
    <div style='background:#FFFBEB; border:1px solid #F59E0B; border-left:4px solid #F59E0B;
                border-radius:10px; padding:1.2rem 1.5rem; margin-bottom:1.5rem;'>
        <div style='color:#92400E; font-weight:600; font-size:0.9rem; margin-bottom:0.5rem;'>
            📋 Contexte d'évaluation — À lire avant d'interpréter ces métriques
        </div>
        <div style='color:#78350F; font-size:0.82rem; line-height:1.8;'>
            Ces performances sont mesurées sur le <strong>test set de PlantVillage</strong>, 
            un dataset en conditions contrôlées (fond neutre, feuille isolée, éclairage uniforme).<br>
            Elles ne reflètent <strong>pas nécessairement</strong> les performances en conditions réelles (photos terrain).<br><br>
            <strong>Limites identifiées :</strong><br>
            • <strong>Domain shift</strong> : écart entre images laboratoire et photos terrain<br>
            • <strong>Espèces limitées</strong> : 9 espèces uniquement (pomme, tomate, maïs…)<br>
            • <strong>Biais dataset</strong> : PlantVillage est reconnu comme trop "propre" dans la littérature<br><br>
            <em>Pour un déploiement en production, un test sur données terrain serait nécessaire.</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Architecture
    col_arch, col_data = st.columns(2, gap="large")

    with col_arch:
        st.markdown("<div class='section-title'>Architecture du Modèle</div>", unsafe_allow_html=True)
        arch_data = {
            "Composant": ["Base Model", "Pooling", "Dense Layer 1", "Batch Norm", "Dropout 1", "Dense Layer 2", "Dropout 2", "Output"],
            "Détail": ["EfficientNetB0 (ImageNet)", "GlobalAveragePooling2D", "256 neurones — ReLU", "Normalisation", "40%", "128 neurones — ReLU", "30%", f"{NUM_CLASSES} classes — Softmax"]
        }
        df_arch = pd.DataFrame(arch_data)
        st.dataframe(df_arch, use_container_width=True)

    with col_data:
        st.markdown("<div class='section-title'>Dataset & Entraînement</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <div class='metric-row'>
                <span class='metric-label'>Dataset</span>
                <span class='metric-value'>PlantVillage (Updated)</span>
            </div>
            <div class='metric-row'>
                <span class='metric-label'>Total images</span>
                <span class='metric-value'>{NUM_IMAGES:,}</span>
            </div>
            <div class='metric-row'>
                <span class='metric-label'>Train / Val / Test</span>
                <span class='metric-value'>70% / 15% / 15%</span>
            </div>
            <div class='metric-row'>
                <span class='metric-label'>Split</span>
                <span class='metric-value'>Stratifié (sklearn)</span>
            </div>
            <div class='metric-row'>
                <span class='metric-label'>Classes</span>
                <span class='metric-value'>{NUM_CLASSES} maladies / espèces</span>
            </div>
            <div class='metric-row'>
                <span class='metric-label'>Phase 1</span>
                <span class='metric-value'>Feature Extraction (10 epochs)</span>
            </div>
            <div class='metric-row'>
                <span class='metric-label'>Phase 2</span>
                <span class='metric-value'>Fine-Tuning (8 epochs)</span>
            </div>
            <div class='metric-row'>
                <span class='metric-label'>Optimizer</span>
                <span class='metric-value'>Adam (lr: 1e-3 → 1e-5)</span>
            </div>
            <div class='metric-row'>
                <span class='metric-label'>Augmentation</span>
                <span class='metric-value'>Rotation, Zoom, Flip, Brightness</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Distribution des classes
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Distribution des Classes</div>", unsafe_allow_html=True)

    healthy = [k for k in CLASS_INFO if CLASS_INFO[k]["status"] == "healthy"]
    diseased = [k for k in CLASS_INFO if CLASS_INFO[k]["status"] == "diseased"]

    fig_pie = go.Figure(data=[go.Pie(
        labels=["Sain", "Malade"],
        values=[len(healthy), len(diseased)],
        hole=0.6,
        marker_colors=["#2D6A4F", "#E07B39"],
        textfont_size=14,
    )])
    fig_pie.update_layout(
        height=300,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        font={'family': 'DM Sans'},
        annotations=[dict(text=f'{NUM_CLASSES}<br>classes', x=0.5, y=0.5,
                         font_size=16, showarrow=False, font_color='#2D6A4F')]
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # ── Courbes d'entraînement ──────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Courbes d'Entraînement</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box' style='font-size:0.82rem; color:#555; margin-bottom:1rem;'>
        <strong>Lecture :</strong> La ligne verticale pointillée marque le début du Fine-Tuning (Phase 2).
        La validation accuracy dépasse la train accuracy en Phase 1 — signe que le modèle généralise bien
        sans overfitting. La légère chute au début du Fine-Tuning est normale (adaptation du learning rate).
    </div>
    """, unsafe_allow_html=True)
    st.image(Image.open(Path(__file__).parent / "training_curves.png"), use_column_width=True)

    # ── Matrice de confusion ──────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Matrice de Confusion — Test Set</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box' style='font-size:0.82rem; color:#555; margin-bottom:1rem;'>
        <strong>Lecture :</strong> La diagonale représente les prédictions correctes.
        Les confusions les plus notables se situent entre maladies de la <strong>tomate</strong>
        (Early Blight / Septoria / Late Blight) — visuellement similaires même pour un expert humain.
        Les espèces comme la cerise, la pêche et le raisin sont quasi-parfaitement classifiées.
    </div>
    """, unsafe_allow_html=True)
    st.image(Image.open(Path(__file__).parent / "confusion_matrix.png"), use_column_width=True)

    # ── Classes les plus difficiles ──────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Analyse par Classe — F1-Score</div>", unsafe_allow_html=True)

    class_f1 = {
        "Corn (Maize) — Cercospora Leaf Spot": 0.96,
        "Corn (Maize) — Northern Leaf Blight": 0.97,
        "Tomato — Early Blight":               0.95,
        "Tomato — Late Blight":                0.97,
        "Tomato — Septoria Leaf Spot":         0.96,
        "Tomato — Bacterial Spot":             0.97,
        "Grape — Black Rot":                   0.99,
        "Grape — Esca (Black Measles)":        0.99,
        "Apple — Apple Scab":                  0.99,
        "Peach — Bacterial Spot":              1.00,
        "Cherry — Powdery Mildew":             1.00,
        "Strawberry — Leaf Scorch":            1.00,
    }

    df_f1 = pd.DataFrame({
        "Classe": list(class_f1.keys()),
        "F1-Score": list(class_f1.values())
    }).sort_values("F1-Score")

    colors = ["#E07B39" if v < 0.97 else "#2D6A4F" for v in df_f1["F1-Score"]]

    fig_bar = go.Figure(go.Bar(
        x=df_f1["F1-Score"],
        y=df_f1["Classe"],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.0%}" for v in df_f1["F1-Score"]],
        textposition='outside'
    ))
    fig_bar.update_layout(
        height=420,
        margin=dict(t=20, b=20, l=20, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0.93, 1.01], tickformat='.0%', gridcolor='#EEE'),
        font={'family': 'DM Sans', 'size': 12},
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("""
    <div class='info-box' style='font-size:0.82rem; color:#555;'>
        🟠 <strong>Classes difficiles</strong> : les maladies de la tomate se ressemblent visuellement
        (taches nécrotiques similaires). C'est une limite inhérente au problème, pas au modèle.<br>
        🟢 <strong>Classes faciles</strong> : cerise, pêche, fraise — symptômes visuellement très distinctifs.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# PAGE 3 — À PROPOS
# ─────────────────────────────────────────
elif page == "À propos":

    st.markdown("""
    <div class='app-header'>
        <div class='app-title'>À propos</div>
        <div class='app-subtitle'>Méthodologie · Technologie · Auteur</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='section-title'>Le Projet</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box' style='line-height: 1.8; font-size: 0.9rem; color: #444;'>
            PlantGuard AI est un système de diagnostic automatique des maladies végétales 
            basé sur la Computer Vision et le Deep Learning.<br><br>
            Le modèle EfficientNetB0, pré-entraîné sur ImageNet, a été fine-tuné sur le 
            dataset PlantVillage pour classifier 29 types de maladies et états de santé 
            sur 9 espèces de plantes différentes.<br><br>
            L'approche Transfer Learning en 2 phases (Feature Extraction puis Fine-Tuning) 
            permet d'atteindre des performances élevées avec un dataset de taille limitée.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-title'>Auteur</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <div style='font-family: DM Serif Display, serif; font-size: 1.4rem; color: #1A2E1A; margin-bottom: 0.3rem;'>
                Brejnev AKOUMANI
            </div>
            <div style='font-size: 0.85rem; color: #2D6A4F; font-weight: 500; margin-bottom: 1rem;'>
                Data Scientist | ML & IA Générative
            </div>
            <div style='font-size: 0.85rem; color: #555; line-height: 1.8;'>
            <br>
                <a href='https://linkedin.com/in/akbrejnev' style='color: #2D6A4F; font-weight: 500;'>
                    LinkedIn
                </a> &nbsp;·&nbsp;
                <a href='https://github.com/akbrejnev200-rgb' style='color: #2D6A4F; font-weight: 500;'>
                    GitHub
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.markdown("<div class='section-title'>Stack Technique</div>", unsafe_allow_html=True)
        tech_stack = {
            "Framework DL": "TensorFlow / Keras",
            "Modèle base": "EfficientNetB0",
            "Prétraitement": "ImageDataGenerator",
            "Évaluation": "Scikit-learn",
            "Interface": "Streamlit",
            "Visualisation": "Plotly",
            "Entraînement": "Local (GPU RTX 4050 Laptop)",
            "Dataset": "PlantVillage (Kaggle)"
        }
        for key, value in tech_stack.items():
            st.markdown(f"""
            <div class='metric-row'>
                <span class='metric-label'>{key}</span>
                <span class='metric-value'>{value}</span>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='section-title'>Références</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box' style='font-size: 0.85rem; color: #555; line-height: 1.8;'>
            <strong>Dataset :</strong><br>
            PlantVillage Dataset (Updated) — Tushar Sharma, Kaggle<br><br>
            <strong>Modèle :</strong><br>
            EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks<br>
            Tan & Le, ICML 2019<br><br>
            <strong>Entraînement :</strong><br>
            Transfer Learning + Fine-Tuning<br>
            Split stratifié 70/15/15 · EarlyStopping · ReduceLROnPlateau
        </div>
        """, unsafe_allow_html=True)
