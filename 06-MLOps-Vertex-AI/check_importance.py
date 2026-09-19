"""Diagnostic rapide : importance des features du modèle XGBoost sauvegardé."""

import joblib
import pandas as pd

from aml_detection.config import MODELS_DIR
from aml_detection.features import FEATURES

m = joblib.load(MODELS_DIR / "xgboost_model.joblib")
s = pd.Series(m.feature_importances_, index=FEATURES).sort_values(ascending=False)

print(s.head(12).to_string())

vol = ["nb_transactions_envoyees", "nb_transactions_recues", "max_transactions_jour"]
print()
print(f"Part des 2 features de volume brut : {s[vol[:2]].sum():.1%}")
print(f"En ajoutant max_transactions_jour  : {s[vol].sum():.1%}")
