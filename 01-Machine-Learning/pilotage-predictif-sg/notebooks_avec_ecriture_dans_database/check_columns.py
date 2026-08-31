import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from database.db_connection import get_engine

engine = get_engine()

df = pd.read_sql("SELECT * FROM taches_clean LIMIT 5", engine)
print("Colonnes de taches_clean :")
for col in df.columns:
    print(f"  - {col}")
