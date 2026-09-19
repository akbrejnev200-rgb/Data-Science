"""
Chargement des données depuis BigQuery.
"""

import pandas as pd
from google.cloud import bigquery

from .config import BQ_DATASET, BQ_TABLE, PROJECT_ID


def load_account_features(
    project_id: str = PROJECT_ID,
    dataset: str = BQ_DATASET,
    table: str = BQ_TABLE,
) -> pd.DataFrame:
    """Charge la table des features par compte depuis BigQuery."""
    client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{project_id}.{dataset}.{table}`"
    return client.query(query).to_dataframe()
