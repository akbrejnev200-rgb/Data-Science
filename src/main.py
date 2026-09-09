"""
Pipeline complet : chargement, split, entraînement, calibration, évaluation, sauvegarde.
"""

from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
)

from data_loading import load_account_features
from features import prepare_features, FEATURES
from train import (
    train_isolation_forest, select_best_depth, train_random_forest,
    select_best_xgboost_params, train_xgboost,
)
from evaluate import evaluate_isolation_forest, calibrate_threshold, evaluate_final

PROJECT_ID = "mlops-vertex-demo"
# Chemin absolu (indépendant du répertoire de lancement) : <racine du repo>/models
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Taux de contamination fixé a priori pour l'Isolation Forest : ordre de grandeur
# (~2 %) des comptes signalés comme suspects dans les portefeuilles AML réels.
# Valeur métier indépendante des labels du jeu courant — contrairement à
# y.mean(), qui suppose la vérité terrain connue et n'existe pas en production.
CONTAMINATION_PRIOR = 0.02


def main():
    print("Chargement des données depuis BigQuery...")
    df = load_account_features(PROJECT_ID)
    print(f"{len(df)} comptes chargés. Suspects : {df['label_laundering'].sum()} "
          f"({df['label_laundering'].mean()*100:.2f}%)\n")

    X, y = prepare_features(df)

    # --- Split 70/15/15 EN PREMIER : aucun ajustement (scaler, modèle) n'a
    #     encore vu les données de validation ou de test ---
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp
    )
    print(f"Train : {len(X_train)} ({y_train.sum()} suspects) | "
          f"Validation : {len(X_val)} ({y_val.sum()} suspects) | "
          f"Test : {len(X_test)} ({y_test.sum()} suspects)\n")

    # --- Scaler ajusté sur train uniquement, appliqué à validation et test ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Isolation Forest (non supervisé) : entraîné sur train, évalué sur test ---
    print("=" * 60)
    print("MODELE 1 : Isolation Forest")
    print("=" * 60)
    iso_forest = train_isolation_forest(X_train_scaled, CONTAMINATION_PRIOR)
    _, report_iso, detection_rate = evaluate_isolation_forest(
        iso_forest, X_test_scaled, y_test
    )
    print(f"Taux de détection des vrais comptes suspects (test) : {detection_rate:.1f}%")
    print(f"Précision suspect : {report_iso['suspect']['precision']:.3f}, "
          f"Rappel suspect : {report_iso['suspect']['recall']:.3f}\n")

    # --- Sélection Random Forest sur validation ---
    print("=" * 60)
    print("MODELE 2 : Random Forest — sélection d'hyperparamètre")
    print("=" * 60)
    best_depth, depth_results = select_best_depth(X_train, y_train, X_val, y_val)
    for depth, auc_pr in depth_results.items():
        print(f"  max_depth={depth} -> AUC-PR (validation) = {auc_pr:.3f}")
    print(f"Meilleur hyperparamètre retenu : max_depth={best_depth}")
    rf_val = train_random_forest(X_train, y_train, max_depth=best_depth)
    rf_val_auc_pr = average_precision_score(y_val, rf_val.predict_proba(X_val)[:, 1])
    print(f"AUC-PR (validation) Random Forest : {rf_val_auc_pr:.3f}\n")

    # --- Sélection XGBoost sur validation ---
    print("=" * 60)
    print("MODELE 3 : XGBoost — sélection d'hyperparamètres")
    print("=" * 60)
    best_xgb_params, xgb_results, scale_pos_weight = select_best_xgboost_params(
        X_train, y_train, X_val, y_val
    )
    for key, auc_pr in xgb_results.items():
        print(f"  {key} -> AUC-PR (validation) = {auc_pr:.3f}")
    print(f"Meilleurs hyperparamètres retenus : {best_xgb_params}")
    xgb_val = train_xgboost(X_train, y_train, best_xgb_params, scale_pos_weight)
    xgb_val_auc_pr = average_precision_score(y_val, xgb_val.predict_proba(X_val)[:, 1])
    print(f"AUC-PR (validation) XGBoost : {xgb_val_auc_pr:.3f}\n")

    # --- Choix du vainqueur SUR VALIDATION (le test reste vierge) ---
    print("=" * 60)
    print("COMPARAISON SUR VALIDATION (AUC-PR)")
    print("=" * 60)
    print(f"Random Forest : {rf_val_auc_pr:.3f}")
    print(f"XGBoost       : {xgb_val_auc_pr:.3f}")
    xgb_gagne = xgb_val_auc_pr > rf_val_auc_pr

    # --- Modèles finaux réentraînés sur train+validation ---
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    rf_final = train_random_forest(X_train_full, y_train_full, max_depth=best_depth)
    xgb_final = train_xgboost(X_train_full, y_train_full, best_xgb_params, scale_pos_weight)

    if xgb_gagne:
        final_model, model_name, model_val_auc_pr = xgb_final, "XGBoost", xgb_val_auc_pr
    else:
        final_model, model_name, model_val_auc_pr = rf_final, "Random Forest", rf_val_auc_pr
    print(f"\n-> {model_name} retenu comme modèle final "
          f"(AUC-PR validation = {model_val_auc_pr:.3f})")

    # --- Seuil recalibré sur validation AVEC le modèle final ---
    # calibrate_threshold est appliqué au modèle réellement déployé (final_model),
    # pas à un proxy entraîné sur train seul. Limite résiduelle assumée : X_val
    # fait partie de l'entraînement de final_model, le seuil est donc un peu
    # optimiste — compromis préférable à un seuil issu d'un modèle différent.
    best_threshold, threshold_table = calibrate_threshold(final_model, X_val, y_val)
    print(f"Seuil optimal (F1, validation) : {best_threshold:.3f}")
    print("Table des compromis rappel/précision :")
    for target_recall, vals in threshold_table.items():
        print(f"  Rappel visé {target_recall} -> seuil={vals['threshold']:.3f}, "
              f"précision={vals['precision']:.3f}, rappel={vals['recall']:.3f}")

    # --- Évaluation finale sur le test : le modèle retenu, une seule fois ---
    print("\n" + "=" * 60)
    print(f"EVALUATION FINALE SUR LE TEST — {model_name.upper()}")
    print("=" * 60)

    result = evaluate_final(final_model, X_test, y_test, threshold=best_threshold)

    # Vue au seuil par défaut : dérivée de result["y_proba"], sans réévaluation.
    y_pred_05 = (result["y_proba"] >= 0.5).astype(int)
    report_05 = classification_report(
        y_test, y_pred_05, target_names=["normal", "suspect"],
        zero_division=0, output_dict=True,
    )
    print("\n--- Seuil par défaut (0.5) ---")
    print(f"Précision suspect : {report_05['suspect']['precision']:.3f}, "
          f"Rappel suspect : {report_05['suspect']['recall']:.3f}")
    print(confusion_matrix(y_test, y_pred_05))

    print(f"\n--- Seuil calibré ({best_threshold:.3f}) ---")
    print(f"Précision suspect : {result['report']['suspect']['precision']:.3f}, "
          f"Rappel suspect : {result['report']['suspect']['recall']:.3f}")
    print(result["confusion_matrix"])

    print(f"\nAUC-ROC (test) : {result['auc_roc']:.3f}")
    print(f"AUC-PR (test) : {result['auc_pr']:.3f}")

    # --- Sauvegarde des artefacts ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso_forest, MODELS_DIR / "isolation_forest_model.joblib")
    joblib.dump(final_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(rf_final, MODELS_DIR / "random_forest_model.joblib")
    joblib.dump(xgb_final, MODELS_DIR / "xgboost_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(best_threshold, MODELS_DIR / "decision_threshold.joblib")
    print(f"\nModèles sauvegardés dans {MODELS_DIR}/")


if __name__ == "__main__":
    main()
