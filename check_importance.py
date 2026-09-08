import sys, joblib, pandas as pd
sys.path.insert(0, "src")
from features import FEATURES

m = joblib.load("models/xgboost_model.joblib")
s = pd.Series(m.feature_importances_, index=FEATURES).sort_values(ascending=False)

print(s.head(12).to_string())

vol = ["nb_transactions_envoyees", "nb_transactions_recues", "max_transactions_jour"]
print()
print("Part des 2 features de volume brut : {:.1%}".format(s[vol[:2]].sum()))
print("En ajoutant max_transactions_jour  : {:.1%}".format(s[vol].sum()))
