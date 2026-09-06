"""
modelisation_retard.py

Port du Bloc 3C du notebook phase3_modelisation.ipynb (Classification du
risque de retard, RandomForest vs DecisionTree), adapte pour lire depuis
features_detail.

FIT UNIQUEMENT : compare Decision Tree/Random Forest, ajuste le seuil de
decision, sauvegarde models/model_classification_retard.pkl. Ne genere
plus le backtest (c'est le role de predict_retard.py, execute separement).

Ameliorations vs notebook d'origine :
- Encodage one-hot supplementaire de Type_Processus (prefixe proc_), en
  plus de Service et Type_Contrat deja encodes dans la version originale.
- Recherche d'hyperparametres sur validation (au lieu de max_depth=6/10
  et min_samples_leaf=50/20 fixes a la main).
- Seuil de decision ajuste sur validation (maximisation du F2, qui priorise
  le rappel : dans un back-office, manquer un vrai retard coute plus cher
  que verifier une fausse alerte) au lieu du
  seuil par defaut 0.5, pour mieux gerer le desequilibre de classes en
  complement de class_weight="balanced".

Prerequis : feature_engineering.py deja execute (features_detail peuplee).

Usage :
    python modelisation_retard.py
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))
from db_connection import get_engine, MODELS_DIR
from modelisation_common import split_temporel

warnings.filterwarnings("ignore")

engine = get_engine()

FEATURES_NUM = [
    "complexite_num", "coeff_productivite_contrat",
    "charge_journaliere_nb_taches", "etp_agent_jour",
    "etp_disponible_service_jour", "charge_par_etp_service",
    "anciennete_agent_mois", "delai_prise_en_charge_j",
    "jour_semaine", "mois", "trimestre", "semaine_du_mois",
    "est_fin_de_mois", "est_fin_trimestre", "est_lundi", "est_vendredi",
    "est_absent", "alerte_surcharge_service",
]


def walk_forward_clf(X, y, model_class, model_kwargs, taille_fenetre=50000, taille_pred=10000):
    accs = []
    i = taille_fenetre
    while i + taille_pred <= len(X):
        X_tr, y_tr = X.iloc[i - taille_fenetre:i], y.iloc[i - taille_fenetre:i]
        X_pr, y_pr = X.iloc[i:i + taille_pred], y.iloc[i:i + taille_pred]
        m = model_class(**model_kwargs)
        m.fit(X_tr, y_tr)
        accs.append(accuracy_score(y_pr, m.predict(X_pr)))
        i += taille_pred
    return accs


def meilleur_seuil(y_true, proba, beta=2):
    # F2 (beta=2) plutot que F1 : dans un back-office, manquer un vrai retard
    # coute plus cher que verifier une fausse alerte -- on priorise donc le
    # rappel deux fois plus que la precision au moment de choisir le seuil.
    precisions, recalls, seuils = precision_recall_curve(y_true, proba)
    beta2 = beta ** 2
    fbetas = (1 + beta2) * precisions * recalls / np.where(
        (beta2 * precisions + recalls) == 0, np.nan, beta2 * precisions + recalls
    )
    fbetas = np.nan_to_num(fbetas)
    idx = np.argmax(fbetas[:-1]) if len(seuils) > 0 else 0
    return (seuils[idx], fbetas[idx]) if len(seuils) > 0 else (0.5, fbetas[-1])


def run():
    print("=" * 60)
    print("PHASE 3 - BLOC 3C : CLASSIFICATION DU RISQUE DE RETARD")
    print("=" * 60)

    print("\nChargement de features_detail...")
    df = pd.read_sql(
        "SELECT * FROM features_detail",
        engine, parse_dates=["Date_Creation"],
    )
    print(f"  {len(df)} lignes | {df.shape[1]} colonnes")

    print("\nConstruction de la cible 'est_en_retard'...")
    seuil_retard_j = df["duree_traitement_reelle_j"].quantile(0.75)
    print(f"  Seuil de retard : duree > {seuil_retard_j:.1f} jours (P75)")
    df["est_en_retard"] = (df["duree_traitement_reelle_j"] > seuil_retard_j).astype(int)

    df_clf = df[df["duree_traitement_reelle_j"].notna()].copy()
    print(f"  Lignes avec duree connue : {len(df_clf)} / {len(df)}")
    print(f"  Taux de retard : {df_clf['est_en_retard'].mean() * 100:.1f}%")

    df_clf["Service_ref"] = df_clf["Service"]
    df_clf = pd.get_dummies(df_clf, columns=["Service", "Type_Contrat", "Type_Processus"],
                             prefix=["svc", "ctr", "proc"])
    features_dummies = [c for c in df_clf.columns if c.startswith(("svc_", "ctr_", "proc_"))]
    FEATURES = [f for f in FEATURES_NUM + features_dummies if f in df_clf.columns]
    TARGET = "est_en_retard"

    cols_id = ["ID_Tache", "Matricule_Agent", "Date_Creation", "Service_ref"]
    df_model = df_clf[FEATURES + [TARGET] + cols_id].dropna().copy()
    df_model = df_model.sort_values("Date_Creation").reset_index(drop=True)
    print(f"  Lignes utilisables apres dropna : {len(df_model)} / {len(df_clf)}")
    print(f"  Nombre de features : {len(FEATURES)}")

    df_train, df_val, df_test = split_temporel(df_model)
    print(f"  Split : train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

    X_train, y_train = df_train[FEATURES], df_train[TARGET]
    X_val, y_val = df_val[FEATURES], df_val[TARGET]
    X_test, y_test = df_test[FEATURES], df_test[TARGET]

    # --- MODELE 1 : DECISION TREE (recherche sur validation) ---
    print("\n" + "-" * 60)
    print("MODELE 1 - DECISION TREE (recherche sur validation)")
    print("-" * 60)
    grille_tree = [
        {"max_depth": md, "min_samples_leaf": msl}
        for md in [4, 6, 8]
        for msl in [20, 50, 100]
    ]
    meilleur_tree, meilleure_acc_tree = None, None
    for params in grille_tree:
        m = DecisionTreeClassifier(class_weight="balanced", random_state=42, **params)
        m.fit(X_train, y_train)
        acc_v = accuracy_score(y_val, m.predict(X_val))
        if meilleure_acc_tree is None or acc_v > meilleure_acc_tree:
            meilleure_acc_tree, meilleur_tree = acc_v, m
    print(f"  Meilleurs hyperparametres : max_depth={meilleur_tree.max_depth}, "
          f"min_samples_leaf={meilleur_tree.min_samples_leaf}")
    model_tree = meilleur_tree
    acc_tree = accuracy_score(y_test, model_tree.predict(X_test))
    print(f"  Accuracy test : {acc_tree:.3f}")

    # --- MODELE 2 : RANDOM FOREST (recherche sur validation) ---
    print("\n" + "-" * 60)
    print("MODELE 2 - RANDOM FOREST (recherche sur validation)")
    print("-" * 60)
    grille_rf = [
        {"n_estimators": ne, "max_depth": md, "min_samples_leaf": msl}
        for ne in [100, 200]
        for md in [8, 12]
        for msl in [20, 50]
    ]
    meilleur_rf, meilleure_acc_rf = None, None
    for params in grille_rf:
        m = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1, **params)
        m.fit(X_train, y_train)
        acc_v = accuracy_score(y_val, m.predict(X_val))
        print(f"    {params} -> accuracy val={acc_v:.3f}")
        if meilleure_acc_rf is None or acc_v > meilleure_acc_rf:
            meilleure_acc_rf, meilleur_rf = acc_v, m
    print(f"  Meilleurs hyperparametres : n_estimators={meilleur_rf.n_estimators}, "
          f"max_depth={meilleur_rf.max_depth}, min_samples_leaf={meilleur_rf.min_samples_leaf}")
    model_rf = meilleur_rf
    acc_rf = accuracy_score(y_test, model_rf.predict(X_test))
    print(f"  Accuracy test : {acc_rf:.3f}")

    # --- WALK-FORWARD VALIDATION ---
    print("\n" + "-" * 60)
    print("WALK-FORWARD VALIDATION (train, fenetre glissante)")
    print("-" * 60)
    accs_tree_wf = walk_forward_clf(
        df_train[FEATURES], df_train[TARGET], DecisionTreeClassifier,
        {"max_depth": model_tree.max_depth, "min_samples_leaf": model_tree.min_samples_leaf,
         "class_weight": "balanced", "random_state": 42},
    )
    accs_rf_wf = walk_forward_clf(
        df_train[FEATURES], df_train[TARGET], RandomForestClassifier,
        {"n_estimators": 100, "max_depth": model_rf.max_depth, "min_samples_leaf": model_rf.min_samples_leaf,
         "class_weight": "balanced", "random_state": 42, "n_jobs": -1},
    )
    print(f"  Decision Tree : accuracy walk-forward moyenne = {np.mean(accs_tree_wf):.3f} "
          f"({len(accs_tree_wf)} fenetres)")
    print(f"  Random Forest : accuracy walk-forward moyenne = {np.mean(accs_rf_wf):.3f} "
          f"({len(accs_rf_wf)} fenetres)")

    # --- SELECTION DU MODELE ---
    print("\n" + "=" * 60)
    print("COMPARAISON DECISION TREE vs RANDOM FOREST (test)")
    print("=" * 60)
    comparaison = pd.DataFrame({
        "Modele": ["Decision Tree", "Random Forest"],
        "Accuracy (test)": [acc_tree, acc_rf],
        "Accuracy (walk-forward)": [np.mean(accs_tree_wf), np.mean(accs_rf_wf)],
    })
    print(comparaison.to_string(index=False))

    if acc_rf >= acc_tree:
        modele_final, model_final, type_final = "Random Forest", model_rf, "random_forest"
    else:
        modele_final, model_final, type_final = "Decision Tree", model_tree, "decision_tree"
    print(f"\n  Modele retenu pour la production : {modele_final}")

    # --- SEUIL DE DECISION AJUSTE (F2 sur validation, priorise le rappel) ---
    print("\n" + "-" * 60)
    print("AJUSTEMENT DU SEUIL DE DECISION (maximisation F2 sur validation)")
    print("-" * 60)
    proba_val = model_final.predict_proba(X_val)[:, 1]
    seuil_optimal, f2_val = meilleur_seuil(y_val, proba_val)
    print(f"  Seuil optimal : {seuil_optimal:.3f} (F2 validation = {f2_val:.3f}, vs 0.5 par defaut)")

    proba_test = model_final.predict_proba(X_test)[:, 1]
    pred_test_defaut = (proba_test >= 0.5).astype(int)
    pred_test_ajuste = (proba_test >= seuil_optimal).astype(int)
    print(f"  F1 test (seuil 0.5)      : {f1_score(y_test, pred_test_defaut):.3f}")
    print(f"  F1 test (seuil ajuste)   : {f1_score(y_test, pred_test_ajuste):.3f}")
    print("\n" + classification_report(y_test, pred_test_ajuste, target_names=["Normal", "En retard"], zero_division=0))

    # --- Export du modele (FIT uniquement) ---
    # Le backtest (predictions_risque_retard) est genere separement par
    # predict_retard.py, qui peut tourner chaque jour sans repasser par les
    # recherches d'hyperparametres ci-dessus. On sauvegarde les dates de
    # coupure train/val/test pour que predict_retard.py etiquette
    # correctement chaque ligne (colonne "split", absente de l'ancienne
    # version -- corrige ici : sans elle, la matrice de confusion affichee
    # sur la page Fiabilite melangeait train/val/test).
    joblib.dump({
        "model": model_final, "features": FEATURES, "seuil_retard_j": seuil_retard_j,
        "seuil_decision": seuil_optimal, "accuracy_test": acc_rf if modele_final == "Random Forest" else acc_tree,
        "type": type_final,
        "date_fin_train": df_train["Date_Creation"].max(),
        "date_fin_val": df_val["Date_Creation"].max(),
    }, MODELS_DIR / "model_classification_retard.pkl")
    print(f"  models/model_classification_retard.pkl sauvegarde (modele retenu : {modele_final})")

    print("\n" + "=" * 60)
    print("RAPPORT FINAL - BLOC 3C")
    print("=" * 60)
    print(f"  Cible : est_en_retard = 1 si duree traitement > {seuil_retard_j:.1f} jours (P75)")
    print(f"  Taux de retard dans les donnees : {df_clf['est_en_retard'].mean() * 100:.1f}%")
    print(f"  Modele retenu : {modele_final}")
    print(f"  Decision Tree -> Accuracy = {acc_tree:.3f}")
    print(f"  Random Forest -> Accuracy = {acc_rf:.3f}")
    print(f"  Seuil de decision optimal : {seuil_optimal:.3f} (F1 = {f1_score(y_test, pred_test_ajuste):.3f})")
    print("\nBloc 3C termine.")


if __name__ == "__main__":
    run()
