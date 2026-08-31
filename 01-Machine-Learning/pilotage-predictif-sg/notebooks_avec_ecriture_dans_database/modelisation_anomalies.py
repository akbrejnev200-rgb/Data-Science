"""
modelisation_anomalies.py

Port du Bloc 3D du notebook phase3_modelisation.ipynb (Detection
d'anomalies, Isolation Forest + SHAP), adapte pour lire depuis
features_detail.

FIT UNIQUEMENT : entraine l'Isolation Forest, calcule l'explicabilite
SHAP (image models/shap_anomalies_summary.png), sauvegarde
models/model_isolation_forest.pkl. Ne genere plus les scores par tache
(c'est le role de predict_anomalies.py, execute separement).

Amelioration vs notebook d'origine : recherche du nombre d'arbres
(n_estimators) par injection d'anomalies synthetiques sur la VALIDATION
(et non le test, pour eviter toute fuite), contamination=0.02 conservee
telle quelle (hypothese metier deja justifiee dans le notebook original).

Prerequis : feature_engineering.py deja execute (features_detail peuplee).

Usage :
    python modelisation_anomalies.py
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))
from db_connection import get_engine, MODELS_DIR
from modelisation_common import split_temporel

warnings.filterwarnings("ignore")

engine = get_engine()

FEATURES = [
    "Temps_Passe_Declare_Min", "Volume_Dossiers", "productivite_dossiers_par_heure",
    "duree_traitement_reelle_j", "delai_prise_en_charge_j",
    "complexite_num", "coeff_productivite_contrat",
]
CONTAMINATION = 0.02


def injecter_anomalies(df_source, scaler, seed=42):
    """Injecte des anomalies synthetiques (productivite x50) pour valider la detection."""
    df_aug = df_source.copy().reset_index(drop=True)
    n_inject = max(10, int(len(df_aug) * 0.01))
    idx_inject = np.random.RandomState(seed).choice(len(df_aug), size=n_inject, replace=False)

    df_aug["anomalie_injectee"] = 0
    df_aug.loc[idx_inject, "Volume_Dossiers"] = df_aug.loc[idx_inject, "Volume_Dossiers"] * 50
    df_aug.loc[idx_inject, "productivite_dossiers_par_heure"] = (
        df_aug.loc[idx_inject, "Volume_Dossiers"] / (df_aug.loc[idx_inject, "Temps_Passe_Declare_Min"] / 60)
    )
    df_aug.loc[idx_inject, "anomalie_injectee"] = 1

    X_aug = scaler.transform(df_aug[FEATURES])
    return df_aug, X_aug, n_inject


def taux_detection(model, df_aug, X_aug):
    pred = model.predict(X_aug)
    df_aug = df_aug.copy()
    df_aug["anomalie_detectee"] = (pred == -1).astype(int)
    return df_aug.loc[df_aug["anomalie_injectee"] == 1, "anomalie_detectee"].mean()


def run():
    print("=" * 60)
    print("PHASE 3 - BLOC 3D : DETECTION D'ANOMALIES (Isolation Forest + SHAP)")
    print("=" * 60)

    print("\nChargement de features_detail...")
    df = pd.read_sql("SELECT * FROM features_detail", engine, parse_dates=["Date_Creation"])
    cols_meta = ["ID_Tache", "Matricule_Agent", "Service", "Date_Creation", "Type_Processus", "Complexite"]
    df_model = df[FEATURES + cols_meta].dropna(subset=FEATURES).copy()
    print(f"  Lignes utilisables apres dropna : {len(df_model)} / {len(df)}")

    df_model = df_model.sort_values("Date_Creation").reset_index(drop=True)
    df_train, df_val, df_test = split_temporel(df_model)
    print(f"  Split : train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[FEATURES])

    # --- Recherche du nombre d'arbres (injection sur validation) ---
    print("\n" + "-" * 60)
    print("RECHERCHE DE n_estimators (injection d'anomalies sur validation)")
    print("-" * 60)
    meilleur_n, meilleur_taux = None, None
    for n_estimators in [100, 200, 300]:
        m = IsolationForest(n_estimators=n_estimators, contamination=CONTAMINATION, random_state=42, n_jobs=-1)
        m.fit(X_train)
        df_aug_val, X_aug_val, n_inject_val = injecter_anomalies(df_val, scaler)
        taux = taux_detection(m, df_aug_val, X_aug_val)
        print(f"    n_estimators={n_estimators} -> taux detection (validation) = {taux * 100:.1f}%")
        if meilleur_taux is None or taux > meilleur_taux:
            meilleur_taux, meilleur_n = taux, n_estimators
    print(f"  -> n_estimators retenu : {meilleur_n}")

    print("\nEntrainement final sur le train set...")
    model_if = IsolationForest(n_estimators=meilleur_n, contamination=CONTAMINATION, random_state=42, n_jobs=-1)
    model_if.fit(X_train)

    # --- Validation par injection sur le TEST (evaluation finale) ---
    print("\n" + "-" * 60)
    print("VALIDATION FINALE PAR INJECTION (test)")
    print("-" * 60)
    df_test_aug, X_test_aug, n_inject = injecter_anomalies(df_test, scaler)
    taux_detection_test = taux_detection(model_if, df_test_aug, X_test_aug)
    print(f"  {n_inject} anomalies synthetiques injectees (productivite x50)")
    print(f"  Taux de detection (test) : {taux_detection_test * 100:.1f}%")

    # --- SHAP ---
    print("\n" + "-" * 60)
    print("SHAP - EXPLICABILITE DES ANOMALIES")
    print("-" * 60)
    try:
        import shap
        echantillon = df_model.sample(n=min(2000, len(df_model)), random_state=42)
        X_echantillon = scaler.transform(echantillon[FEATURES])
        print("  Calcul des valeurs SHAP sur un echantillon de 2000 lignes...")
        explainer = shap.TreeExplainer(model_if)
        shap_values = explainer.shap_values(X_echantillon)
        shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES).sort_values(ascending=False)
        print("\n  Importance moyenne des features (SHAP) :")
        print(shap_importance.to_string())

        shap.summary_plot(shap_values, X_echantillon, feature_names=FEATURES, show=False)
        plt.tight_layout()
        MODELS_DIR.mkdir(exist_ok=True)
        plt.savefig(MODELS_DIR / "shap_anomalies_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  models/shap_anomalies_summary.png sauvegarde")

        # Export brut (feature, importance moyenne |SHAP|) pour que l'appli
        # Dash puisse tracer son propre graphique lisible (barres triees,
        # libelles francais) plutot que d'incorporer l'image beeswarm brute,
        # pensee pour un public data-scientist (cf. page Derive individuelle).
        shap_importance.rename("importance_moyenne_abs").rename_axis("feature").reset_index().to_csv(
            MODELS_DIR / "shap_anomalies_importance.csv", index=False,
        )
        print("  models/shap_anomalies_importance.csv sauvegarde")
        top_feature_shap = shap_importance.index[0]
    except Exception as e:
        print(f"  SHAP indisponible ({e}), etape ignoree.")
        top_feature_shap = "N/A"

    # --- Export du modele (FIT uniquement) ---
    # Les scores par tache (anomalies_detectees) sont generes separement
    # par predict_anomalies.py, qui peut tourner chaque jour sans repasser
    # par la recherche de n_estimators ni le calcul SHAP (couteux, lie au
    # modele et non aux donnees du jour).
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "model": model_if, "scaler": scaler, "features": FEATURES,
        "n_estimators": meilleur_n, "contamination": CONTAMINATION,
        "taux_detection_test": taux_detection_test,
    }, MODELS_DIR / "model_isolation_forest.pkl")
    print("  models/model_isolation_forest.pkl sauvegarde")

    print("\n" + "=" * 60)
    print("RAPPORT FINAL - BLOC 3D")
    print("=" * 60)
    print(f"  n_estimators retenu (validation) : {meilleur_n}")
    print(f"  Contamination configuree : {CONTAMINATION * 100:.0f}%")
    print(f"  Taux de detection (injection test) : {taux_detection_test * 100:.1f}%")
    print(f"  Top feature explicative (SHAP) : {top_feature_shap}")
    print("  Modele sauvegarde -- lancer predict_anomalies.py pour generer/rafraichir anomalies_detectees")
    print("\nBloc 3D termine.")


if __name__ == "__main__":
    run()
