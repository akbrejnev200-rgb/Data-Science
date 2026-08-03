"""
Pipeline complet : chargement, split, entraînement, calibration, évaluation, sauvegarde.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loading import load_account_features
from features import prepare_features, FEATURES
from train import (
    train_isolation_forest, select_best_depth, train_random_forest,
    select_best_xgboost_params, train_xgboost,
)
from evaluate import evaluate_isolation_forest, calibrate_threshold, evaluate_final

PROJECT_ID = "mlops-vertex-demo"
MODELS_DIR = "../models"


def main():
    print("Chargement des données depuis BigQuery...")
    df = load_account_features(PROJECT_ID)
    print(f"{len(df)} comptes chargés. Suspects : {df['label_laundering'].sum()} "
          f"({df['label_laundering'].mean()*100:.2f}%)\n")

    X, y = prepare_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Isolation Forest (non supervisé, sur l'ensemble des données) ---
    print("=" * 60)
    print("MODELE 1 : Isolation Forest")
    print("=" * 60)
    contamination_rate = y.mean()
    iso_forest = train_isolation_forest(X_scaled, contamination_rate)
    _, report_iso, detection_rate = evaluate_isolation_forest(iso_forest, X_scaled, y)
    print(f"Taux de détection des vrais comptes suspects : {detection_rate:.1f}%")
    print(f"Précision suspect : {report_iso['suspect']['precision']:.3f}, "
          f"Rappel suspect : {report_iso['suspect']['recall']:.3f}\n")

    # --- Split 70/15/15 ---
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp
    )
    print(f"Train : {len(X_train)} ({y_train.sum()} suspects) | "
          f"Validation : {len(X_val)} ({y_val.sum()} suspects) | "
          f"Test : {len(X_test)} ({y_test.sum()} suspects)\n")

    # --- Sélection du meilleur hyperparamètre sur validation ---
    print("=" * 60)
    print("MODELE 2 : Random Forest — sélection d'hyperparamètre")
    print("=" * 60)
    best_depth, depth_results = select_best_depth(X_train, y_train, X_val, y_val)
    for depth, auc_pr in depth_results.items():
        print(f"  max_depth={depth} -> AUC-PR (validation) = {auc_pr:.3f}")
    print(f"Meilleur hyperparamètre retenu : max_depth={best_depth}\n")

    # --- Calibration du seuil sur validation ---
    rf_val = train_random_forest(X_train, y_train, max_depth=best_depth)
    best_threshold, threshold_table = calibrate_threshold(rf_val, X_val, y_val)
    print(f"Seuil optimal (F1) : {best_threshold:.3f}")
    print("Table des compromis rappel/précision :")
    for target_recall, vals in threshold_table.items():
        print(f"  Rappel visé {target_recall} -> seuil={vals['threshold']:.3f}, "
              f"précision={vals['precision']:.3f}, rappel={vals['recall']:.3f}")

    # --- Entraînement final Random Forest sur train+validation, évaluation sur test ---
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    rf_final = train_random_forest(X_train_full, y_train_full, max_depth=best_depth)

    print("\n" + "=" * 60)
    print("EVALUATION FINALE SUR LE TEST — RANDOM FOREST")
    print("=" * 60)

    result_default = evaluate_final(rf_final, X_test, y_test, threshold=0.5)
    print("\n--- Seuil par défaut (0.5) ---")
    print(f"Précision suspect : {result_default['report']['suspect']['precision']:.3f}, "
          f"Rappel suspect : {result_default['report']['suspect']['recall']:.3f}")
    print(result_default["confusion_matrix"])

    result_calibre = evaluate_final(rf_final, X_test, y_test, threshold=best_threshold)
    print(f"\n--- Seuil calibré ({best_threshold:.3f}) ---")
    print(f"Précision suspect : {result_calibre['report']['suspect']['precision']:.3f}, "
          f"Rappel suspect : {result_calibre['report']['suspect']['recall']:.3f}")
    print(result_calibre["confusion_matrix"])

    print(f"\nAUC-ROC (test) : {result_default['auc_roc']:.3f}")
    print(f"AUC-PR (test) : {result_default['auc_pr']:.3f}")

    # ======================================================
    # XGBoost — comparaison
    # ======================================================
    print("\n" + "=" * 60)
    print("MODELE 3 : XGBoost — sélection d'hyperparamètres")
    print("=" * 60)
    best_xgb_params, xgb_results, scale_pos_weight = select_best_xgboost_params(
        X_train, y_train, X_val, y_val
    )
    for key, auc_pr in xgb_results.items():
        print(f"  {key} -> AUC-PR (validation) = {auc_pr:.3f}")
    print(f"Meilleurs hyperparamètres retenus : {best_xgb_params}")

    xgb_val = train_xgboost(X_train, y_train, best_xgb_params, scale_pos_weight)
    best_threshold_xgb, threshold_table_xgb = calibrate_threshold(xgb_val, X_val, y_val)
    print(f"\nSeuil optimal XGBoost (F1) : {best_threshold_xgb:.3f}")
    print("Table des compromis rappel/précision (XGBoost) :")
    for target_recall, vals in threshold_table_xgb.items():
        print(f"  Rappel visé {target_recall} -> seuil={vals['threshold']:.3f}, "
              f"précision={vals['precision']:.3f}, rappel={vals['recall']:.3f}")

    xgb_final = train_xgboost(X_train_full, y_train_full, best_xgb_params, scale_pos_weight)

    print("\n" + "=" * 60)
    print("EVALUATION FINALE SUR LE TEST — XGBOOST")
    print("=" * 60)

    xgb_result_default = evaluate_final(xgb_final, X_test, y_test, threshold=0.5)
    print("\n--- Seuil par défaut (0.5) ---")
    print(f"Précision suspect : {xgb_result_default['report']['suspect']['precision']:.3f}, "
          f"Rappel suspect : {xgb_result_default['report']['suspect']['recall']:.3f}")
    print(xgb_result_default["confusion_matrix"])

    xgb_result_calibre = evaluate_final(xgb_final, X_test, y_test, threshold=best_threshold_xgb)
    print(f"\n--- Seuil calibré ({best_threshold_xgb:.3f}) ---")
    print(f"Précision suspect : {xgb_result_calibre['report']['suspect']['precision']:.3f}, "
          f"Rappel suspect : {xgb_result_calibre['report']['suspect']['recall']:.3f}")
    print(xgb_result_calibre["confusion_matrix"])

    print(f"\nAUC-ROC (test) : {xgb_result_default['auc_roc']:.3f}")
    print(f"AUC-PR (test) : {xgb_result_default['auc_pr']:.3f}")

    print("\n" + "=" * 60)
    print("COMPARAISON FINALE Random Forest vs XGBoost (AUC-PR test)")
    print("=" * 60)
    print(f"Random Forest : {result_default['auc_pr']:.3f}")
    print(f"XGBoost       : {xgb_result_default['auc_pr']:.3f}")

    # --- Sauvegarde des artefacts (on garde le meilleur des deux comme modèle principal) ---
    if xgb_result_default['auc_pr'] > result_default['auc_pr']:
        meilleur_modele = xgb_final
        meilleur_seuil = best_threshold_xgb
        print("\n-> XGBoost retenu comme modèle final")
    else:
        meilleur_modele = rf_final
        meilleur_seuil = best_threshold
        print("\n-> Random Forest retenu comme modèle final")

    joblib.dump(iso_forest, f"{MODELS_DIR}/isolation_forest_model.joblib")
    joblib.dump(meilleur_modele, f"{MODELS_DIR}/best_model.joblib")
    joblib.dump(rf_final, f"{MODELS_DIR}/random_forest_model.joblib")
    joblib.dump(xgb_final, f"{MODELS_DIR}/xgboost_model.joblib")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.joblib")
    joblib.dump(meilleur_seuil, f"{MODELS_DIR}/decision_threshold.joblib")
    print(f"\nModèles sauvegardés dans {MODELS_DIR}/")


if __name__ == "__main__":
    main()
