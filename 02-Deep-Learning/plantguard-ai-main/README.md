# 🌿 PlantGuard AI — Détection de Maladies des Plantes

> Système de diagnostic des maladies végétales par intelligence artificielle  
> **Auteur : Brejnev AKOUMANI**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://plantguard-ai-1.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-98.94%25-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Description

PlantGuard AI est une application de Computer Vision qui détecte automatiquement
les maladies des plantes à partir d'une photo de feuille.

Le modèle est basé sur **EfficientNetB0** avec Transfer Learning et Fine-Tuning
en 2 phases, entraîné sur le dataset **PlantVillage (Updated)**.

---

## 🎯 Performances

| Métrique  | Score  |
|-----------|--------|
| Accuracy  | 98.94% |
| F1-Score  | 0.99   |
| Précision | 0.99   |
| Rappel    | 0.99   |

> ⚠️ **Note** : Ces performances sont mesurées sur le test set de PlantVillage
> (conditions contrôlées). En conditions réelles (photos terrain), un domain shift
> peut réduire les performances. Un seuil de confiance de 50% minimum est appliqué
> pour éviter les faux diagnostics.

---

## 🌱 Espèces supportées

| Espèce | Classes |
|--------|---------|
| 🍎 Pomme | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| 🌶️ Poivron | Bacterial Spot, Healthy |
| 🍒 Cerise | Powdery Mildew, Healthy |
| 🌽 Maïs | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🍇 Raisin | Black Rot, Esca, Leaf Blight, Healthy |
| 🍑 Pêche | Bacterial Spot, Healthy |
| 🥔 Pomme de terre | Early Blight, Late Blight, Healthy |
| 🍓 Fraise | Leaf Scorch, Healthy |
| 🍅 Tomate | Bacterial Spot, Early Blight, Late Blight, Septoria, Yellow Leaf Curl, Healthy |

---

## 🏗️ Architecture du Modèle

```
EfficientNetB0 (ImageNet) — couches gelées
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu) → BatchNorm → Dropout(0.4)
    ↓
Dense(128, relu) → Dropout(0.3)
    ↓
Dense(29, softmax)
```

**Phase 1 — Feature Extraction** (10 epochs, lr=1e-4)  
**Phase 2 — Fine-Tuning** (8 epochs, lr=1e-5, 10 dernières couches dégelées)

---

## 🚀 Installation & Lancement

### Prérequis
- Python 3.10
- GPU recommandé (CUDA 11.x)

### Installation

```bash
# Cloner le repo
git clone https://github.com/ton-username/plantguard-ai.git
cd plantguard-ai

# Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

### Télécharger le modèle

Le modèle (>100MB) n'est pas inclus dans le repo.  
Télécharge-le depuis [Google Drive](LIEN_A_AJOUTER) et place le dossier `saved_model/` à la racine.

```
plantguard-ai/
├── app.py
├── requirements.txt
├── README.md
├── saved_model/        ← à télécharger séparément
│   ├── saved_model.pb
│   └── variables/
├── training_curves.png
└── confusion_matrix.png
```

### Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`

---

## 📊 Dataset

- **Source** : [PlantVillage Dataset (Updated)](https://www.kaggle.com/datasets/tushar5harma/plant-village-dataset-updated)
- **Images** : 67 111 images
- **Classes** : 29
- **Split** : 70% train / 15% val / 15% test (stratifié)

---

## 📁 Structure du Projet

```
plantguard-ai/
├── app.py                  # Application Streamlit
├── plant_disease_vscode.py # Script d'entraînement
├── requirements.txt        # Dépendances
├── training_curves.png     # Courbes d'entraînement
├── confusion_matrix.png    # Matrice de confusion
└── README.md
```

---

## 🔬 Résultats détaillés

Les classes les plus difficiles sont les maladies de la tomate
(Early Blight / Septoria / Late Blight) dont les symptômes visuels
sont très similaires — même pour un expert humain.

| Classe | F1-Score |
|--------|----------|
| Tomato — Early Blight | 0.95 |
| Tomato — Septoria Leaf Spot | 0.96 |
| Corn — Cercospora Leaf Spot | 0.96 |
| Strawberry — Leaf Scorch | 1.00 |
| Cherry — Powdery Mildew | 1.00 |

---

