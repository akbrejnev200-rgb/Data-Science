"""
Chargement des données depuis BigQuery.
"""

from google.cloud import bigquery


def load_account_features(project_id: str) -> "pd.DataFrame":
    """Charge la table account_features depuis BigQuery."""
    client = bigquery.Client(project=project_id)
    query = f"""
    SELECT *
    FROM `{project_id}.aml_detection.account_features`
    """
    df = client.query(query).to_dataframe()
    return df
