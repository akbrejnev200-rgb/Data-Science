import joblib
import os

model_path = os.path.join("..", "models", "model_classification_retard.pkl")
obj = joblib.load(model_path)
print("Type:", type(obj))
if isinstance(obj, dict):
    print("Cles du dictionnaire:")
    for k, v in obj.items():
        print(f"  - {k}: {type(v)}")
