import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────
# CHEMINS RELATIFS — fonctionne sur Windows, Mac, Linux
# Tous les fichiers doivent être dans le MÊME dossier que app.py
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_LOAN_RAW   = os.path.join(BASE_DIR, "loan_data.csv")
PATH_LOAN_CLEAN = os.path.join(BASE_DIR, "loan_data_clean.csv")
PATH_SCALER     = os.path.join(BASE_DIR, "scaler.pkl")
PATH_LR         = os.path.join(BASE_DIR, "logistic_regression.pkl")
PATH_RF         = os.path.join(BASE_DIR, "random_forest.pkl")
PATH_META       = os.path.join(BASE_DIR, "metadata.json")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION DE LA PAGE
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prédiction de Prêt Bancaire",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .app-logo {
        background: linear-gradient(135deg, #0d47a1, #1976d2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    .app-logo h2 { margin: 0; font-size: 1.3rem; }
    .app-logo p  { margin: 0.2rem 0 0; font-size: 0.8rem; opacity: 0.85; }

    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0d47a1;
        border-left: 4px solid #1976d2;
        padding-left: 0.6rem;
        margin: 1.2rem 0 0.8rem;
    }
    .result-box {
        border-radius: 12px;
        padding: 1.4rem;
        text-align: center;
        margin: 1rem 0;
    }
    .approved { background: #e8f5e9; border: 2px solid #4caf50; }
    .rejected { background: #ffebee; border: 2px solid #f44336; }
    .result-box h2 { margin: 0 0 0.3rem; font-size: 2rem; }
    .result-box p  { margin: 0; color: #555; }

    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-top: 4px solid #1976d2;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .kpi-card .val  { font-size: 1.8rem; font-weight: bold; color: #0d47a1; }
    .kpi-card .lbl  { font-size: 0.8rem; color: #777; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES (avec cache)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    df_raw   = pd.read_csv(PATH_LOAN_RAW)
    df_clean = pd.read_csv(PATH_LOAN_CLEAN)
    return df_raw, df_clean

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES MODÈLES (avec cache)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler   = joblib.load(PATH_SCALER)
    model_lr = joblib.load(PATH_LR)
    model_rf = joblib.load(PATH_RF)
    with open(PATH_META, encoding='utf-8') as f:
        meta = json.load(f)
    return scaler, model_lr, model_rf, meta

df_raw, df_clean = load_dataset()
scaler, model_lr, model_rf, meta = load_models()
feature_names = meta['feature_names']
FEATURES = [f for f in feature_names if f != 'Loan_Status']

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="app-logo">
        <h2>🏦 LoanPredict AI</h2>
        <p>Système de décision bancaire</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 Modèle actif")
    model_choice = st.selectbox(
        "Choisir le modèle",
        ["Logistic Regression", "Random Forest"],
        label_visibility="collapsed"
    )
    model_key    = "logistic_regression" if model_choice == "Logistic Regression" else "random_forest"
    active_model = model_lr if model_key == "logistic_regression" else model_rf

    m = meta['models'][model_key]
    st.markdown(f"""
    <div style="background:#f0f4ff;padding:0.9rem;border-radius:8px;font-size:0.88rem">
        <b>📊 Performances (test set)</b><br><br>
        🎯 Accuracy&nbsp;&nbsp;: <b>{m['accuracy']:.2%}</b><br>
        🔍 Precision : <b>{m['precision']:.2%}</b><br>
        📡 Recall&nbsp;&nbsp;&nbsp;&nbsp;: <b>{m['recall']:.2%}</b><br>
        ⚖️ F1-Score&nbsp; : <b>{m['f1']:.2%}</b><br>
        📈 AUC&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <b>{m['auc']:.4f}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📁 Dataset")
    st.markdown(f"""
    - **Lignes :** {len(df_raw)}
    - **Features :** {meta['n_features']}
    - **Train / Test :** {meta['n_train']} / {meta['n_test']}
    - **Cible :** Loan_Status (Y/N)
    """)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_explore, tab_predict, tab_perf = st.tabs([
    "📊 Exploration des données",
    "🤖 Prédiction",
    "📈 Performance du modèle"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — EXPLORATION DES DONNÉES
# ══════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("## 📊 Exploration des données")
    st.markdown("Analyse du dataset brut et des distributions des variables.")

    n_approved = (df_raw['Loan_Status'] == 'Y').sum()
    n_rejected = (df_raw['Loan_Status'] == 'N').sum()
    n_missing  = df_raw.isnull().sum().sum()
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, len(df_raw),  "Total observations"),
        (c2, n_approved,   "Prêts approuvés ✅"),
        (c3, n_rejected,   "Prêts refusés ❌"),
        (c4, n_missing,    "Valeurs manquantes"),
    ]:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="val">{val}</div>
            <div class="lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-title">📋 Dataset brut</div>', unsafe_allow_html=True)
    nb_rows = st.slider("Nombre de lignes à afficher", 5, 50, 10)
    st.dataframe(df_raw.head(nb_rows), use_container_width=True)

    st.markdown('<div class="section-title">⚠️ Valeurs manquantes</div>', unsafe_allow_html=True)
    missing = df_raw.isnull().sum().reset_index()
    missing.columns = ['Colonne', 'Manquantes']
    missing['%'] = (missing['Manquantes'] / len(df_raw) * 100).round(2)
    missing = missing[missing['Manquantes'] > 0].sort_values('Manquantes', ascending=False)
    col_mv1, col_mv2 = st.columns([1, 2])
    with col_mv1:
        st.dataframe(missing, use_container_width=True, hide_index=True)
    with col_mv2:
        fig_mv = px.bar(missing, x='Colonne', y='Manquantes', text='%',
                        color='Manquantes', color_continuous_scale='Reds',
                        title="Valeurs manquantes par colonne")
        fig_mv.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_mv.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_mv, use_container_width=True)

    st.markdown('<div class="section-title">📈 Distributions</div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        vc = df_raw['Loan_Status'].value_counts().reset_index()
        vc.columns = ['Statut', 'Nb']
        vc['Statut'] = vc['Statut'].map({'Y': 'Approuvé ✅', 'N': 'Refusé ❌'})
        fig_pie = px.pie(vc, values='Nb', names='Statut',
                         color='Statut',
                         color_discrete_map={'Approuvé ✅': '#4caf50', 'Refusé ❌': '#f44336'},
                         title="Répartition Loan_Status")
        fig_pie.update_layout(height=320)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_d2:
        num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        var_choice = st.selectbox("Variable numérique", num_cols)
        fig_hist = px.histogram(df_raw, x=var_choice, color='Loan_Status',
                                color_discrete_map={'Y': '#4caf50', 'N': '#f44336'},
                                barmode='overlay', opacity=0.7,
                                title=f"Distribution de {var_choice} par statut")
        fig_hist.update_layout(height=320)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown('<div class="section-title">📊 Variables catégorielles vs Loan_Status</div>', unsafe_allow_html=True)
    cat_cols   = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History', 'Dependents']
    cat_choice = st.selectbox("Variable catégorielle", cat_cols)
    df_cat = df_raw.groupby([cat_choice, 'Loan_Status']).size().reset_index(name='count')
    fig_cat = px.bar(df_cat, x=cat_choice, y='count', color='Loan_Status',
                     color_discrete_map={'Y': '#4caf50', 'N': '#f44336'},
                     barmode='group', title=f"{cat_choice} vs Loan_Status")
    fig_cat.update_layout(height=380)
    st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown('<div class="section-title">🔗 Matrice de corrélation</div>', unsafe_allow_html=True)
    corr = df_clean.corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                         aspect='auto', title="Corrélations entre variables")
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — PRÉDICTION
# ══════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown(f"## 🤖 Prédiction — *{model_choice}*")

    col_form, col_res = st.columns([1.1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-title">👤 Profil du demandeur</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            gender    = st.selectbox("Genre", ["Male", "Female"])
            married   = st.selectbox("Marié(e)", ["Yes", "No"])
            education = st.selectbox("Niveau d'études", ["Graduate", "Not Graduate"])
        with c2:
            dependents    = st.number_input("Dépendants", 0, 5, 0)
            self_employed = st.selectbox("Indépendant", ["No", "Yes"])
            property_area = st.selectbox("Zone propriété", ["Urban", "Semiurban", "Rural"])

        st.markdown('<div class="section-title">💰 Informations financières</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            applicant_income   = st.number_input("Revenu demandeur (€/mois)", 0, 200000, 5000, 500)
            coapplicant_income = st.number_input("Revenu co-demandeur (€/mois)", 0, 100000, 0, 500)
        with c4:
            loan_amount = st.number_input("Montant du prêt (k€)", 10, 1000, 150, 10)
            loan_term   = st.selectbox("Durée (mois)", [360, 300, 240, 180, 120, 84, 60, 36, 12])

        credit_history = st.radio(
            "Historique de crédit",
            [1, 0],
            format_func=lambda x: "✅ Bon historique" if x == 1 else "❌ Mauvais historique",
            horizontal=True
        )
        predict_btn = st.button("🔮 Lancer la prédiction", type="primary", use_container_width=True)

    with col_res:
        if predict_btn:
            total_income  = applicant_income + coapplicant_income
            emi_to_income = (loan_amount * 1000) / (loan_term * (total_income + 1))

            row = {
                'Gender':                  1 if gender == "Male" else 0,
                'Married':                 1 if married == "Yes" else 0,
                'Dependents':              int(dependents),
                'Education':               1 if education == "Graduate" else 0,
                'Self_Employed':           1 if self_employed == "Yes" else 0,
                'Loan_Amount_Term':        float(loan_term),
                'Credit_History':          float(credit_history),
                'EMI_to_Income':           emi_to_income,
                'LoanAmount_log':          np.log1p(loan_amount),
                'TotalIncome_log':         np.log1p(total_income),
                'AppIncome_log':           np.log1p(applicant_income),
                'Property_Area_Rural':     1 if property_area == "Rural" else 0,
                'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
                'Property_Area_Urban':     1 if property_area == "Urban" else 0,
            }
            X_in = pd.DataFrame([row])[FEATURES]

            if model_key == "logistic_regression":
                X_sc  = scaler.transform(X_in)
                pred  = model_lr.predict(X_sc)[0]
                proba = model_lr.predict_proba(X_sc)[0]
            else:
                pred  = model_rf.predict(X_in)[0]
                proba = model_rf.predict_proba(X_in)[0]

            p_approved = proba[1]

            if pred == 1:
                st.markdown(f"""<div class="result-box approved">
                    <h2>✅ PRÊT APPROUVÉ</h2>
                    <p>Probabilité d'approbation : <b>{p_approved:.1%}</b></p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-box rejected">
                    <h2>❌ PRÊT REFUSÉ</h2>
                    <p>Probabilité d'approbation : <b>{p_approved:.1%}</b></p>
                </div>""", unsafe_allow_html=True)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=p_approved * 100,
                number={'suffix': '%'},
                title={'text': "Probabilité d'approbation"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#4caf50" if pred == 1 else "#f44336"},
                    'steps': [
                        {'range': [0,  40], 'color': '#ffcdd2'},
                        {'range': [40, 65], 'color': '#fff9c4'},
                        {'range': [65,100], 'color': '#c8e6c9'},
                    ],
                    'threshold': {'line': {'color':'#333','width':3}, 'thickness':0.75, 'value':50}
                }
            ))
            fig_g.update_layout(height=270, margin=dict(t=50, b=10, l=20, r=20))
            st.plotly_chart(fig_g, use_container_width=True)

            ca, cr = st.columns(2)
            ca.metric("✅ Prob. Approuvé", f"{proba[1]:.1%}")
            cr.metric("❌ Prob. Refusé",   f"{proba[0]:.1%}")

            st.markdown("**💡 Facteurs analysés :**")
            if credit_history == 1: st.success("Bon historique de crédit")
            else: st.error("Mauvais historique de crédit")
            if emi_to_income < 0.3: st.success(f"Ratio EMI/Revenu acceptable ({emi_to_income:.2f})")
            else: st.warning(f"Ratio EMI/Revenu élevé ({emi_to_income:.2f})")
            if total_income > 5000: st.success(f"Revenu total satisfaisant ({total_income:,} €)")
            else: st.warning(f"Revenu total faible ({total_income:,} €)")
            if property_area == "Semiurban": st.success("Zone semi-urbaine (favorable)")
        else:
            st.info("👈 Remplissez le formulaire et cliquez sur **Lancer la prédiction**")
            st.markdown("""
            #### Comment ça fonctionne ?
            1. Saisissez les informations du demandeur
            2. Choisissez le modèle dans la barre latérale
            3. Cliquez sur **Lancer la prédiction**
            4. Obtenez la décision et la probabilité en temps réel
            """)


# ══════════════════════════════════════════════════════════════
# TAB 3 — PERFORMANCE DU MODÈLE
# ══════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown("## 📈 Performance du modèle")
    st.markdown(f"Résultats détaillés pour : **{model_choice}**")

    mlr = meta['models']['logistic_regression']
    mrf = meta['models']['random_forest']
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    labels_fr    = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']

    st.markdown('<div class="section-title">⚖️ Comparaison LR vs RF</div>', unsafe_allow_html=True)
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name='Logistic Regression', x=labels_fr,
        y=[mlr[k] for k in metrics_keys],
        marker_color='#1565c0',
        text=[f"{mlr[k]:.3f}" for k in metrics_keys], textposition='outside'
    ))
    fig_cmp.add_trace(go.Bar(
        name='Random Forest', x=labels_fr,
        y=[mrf[k] for k in metrics_keys],
        marker_color='#2e7d32',
        text=[f"{mrf[k]:.3f}" for k in metrics_keys], textposition='outside'
    ))
    fig_cmp.update_layout(barmode='group', yaxis_range=[0, 1.15],
                          height=420, legend=dict(orientation='h', y=1.05))
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown('<div class="section-title">🔲 Matrice de Confusion — Modèle actif</div>', unsafe_allow_html=True)
    cm = np.array(meta['models'][model_key]['confusion_matrix'])
    col_cm, col_cl = st.columns(2)
    with col_cm:
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           x=['Prédit Refusé', 'Prédit Approuvé'],
                           y=['Vrai Refusé', 'Vrai Approuvé'],
                           title=f"Matrice de confusion — {model_choice}")
        fig_cm.update_layout(height=350)
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_cl:
        TN, FP, FN, TP = cm.ravel()
        total = cm.sum()
        st.markdown(f"""
        <div style="padding:1rem;background:#f8f9fa;border-radius:10px;font-size:0.9rem">
            <b>🧮 Analyse détaillée</b><br><br>
            ✅ <b>Vrais Positifs (TP)</b> : {TP} — Prêts bien approuvés<br>
            ✅ <b>Vrais Négatifs (TN)</b> : {TN} — Prêts bien refusés<br>
            ⚠️ <b>Faux Positifs (FP)</b> : {FP} — Approuvés à tort (risque banque)<br>
            ⚠️ <b>Faux Négatifs (FN)</b> : {FN} — Refusés à tort (clients perdus)<br><br>
            📊 <b>Erreurs totales</b> : {FP+FN} ({(FP+FN)/total*100:.1f}%)<br>
            📊 <b>Bonnes prédictions</b> : {TP+TN} ({(TP+TN)/total*100:.1f}%)
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔍 Feature Importance — Modèle actif</div>', unsafe_allow_html=True)
    fi = meta['models'][model_key]['feature_importance']
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    nice_labels = {
        'Credit_History': 'Credit History',
        'TotalIncome_log': 'Total Income (log)',
        'LoanAmount_log': 'Loan Amount (log)',
        'AppIncome_log': 'Applicant Income (log)',
        'EMI_to_Income': 'EMI / Income ratio',
        'Loan_Amount_Term': 'Loan Term',
        'Property_Area_Semiurban': 'Zone Semi-urbaine',
        'Property_Area_Rural': 'Zone Rurale',
        'Property_Area_Urban': 'Zone Urbaine',
        'Married': 'Married',
        'Education': 'Education',
        'Gender': 'Gender',
        'Dependents': 'Dependents',
        'Self_Employed': 'Self Employed',
    }
    labels = [nice_labels.get(k, k) for k in fi_sorted.keys()]
    values = list(fi_sorted.values())
    fig_fi = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker_color=['#1565c0' if model_key == 'logistic_regression' else '#2e7d32'] * len(values),
        text=[f"{v:.3f}" for v in values], textposition='outside'
    ))
    fig_fi.update_layout(title=f"Importance des variables — {model_choice}",
                         xaxis_title="Importance", height=480, margin=dict(l=160))
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="section-title">📋 Tableau récapitulatif</div>', unsafe_allow_html=True)
    df_tbl = pd.DataFrame({
        'Métrique': labels_fr + ['CV AUC (mean)', 'CV AUC (std)'],
        'Logistic Regression': [f"{mlr[k]:.4f}" for k in metrics_keys] + [f"{mlr['cv_auc_mean']:.4f}", f"{mlr['cv_auc_std']:.4f}"],
        'Random Forest':       [f"{mrf[k]:.4f}" for k in metrics_keys] + [f"{mrf['cv_auc_mean']:.4f}", f"{mrf['cv_auc_std']:.4f}"],
        'Meilleur': ["🏆 LR" if mlr[k] >= mrf[k] else "🏆 RF" for k in metrics_keys] + ["—", "—"]
    })
    st.dataframe(df_tbl, use_container_width=True, hide_index=True)

    best = "Logistic Regression" if mlr['auc'] >= mrf['auc'] else "Random Forest"
    st.info(f"💡 **Recommandation :** **{best}** obtient le meilleur AUC. "
            "La Régression Logistique reste préférable en contexte bancaire pour son interprétabilité.")
