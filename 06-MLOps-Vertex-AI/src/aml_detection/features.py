"""
Définition et préparation des features utilisées par les modèles.
"""

import numpy as np

FEATURES = [
    "nb_transactions_envoyees", "nb_transactions_recues",
    "montant_total_envoye", "montant_total_recu",
    "montant_moyen_envoye", "montant_moyen_recu",
    "montant_max_envoye", "montant_max_recu",
    "nb_devises_envoi", "nb_devises_reception",
    "nb_banques_destinataires", "nb_banques_expediteurs",
    "nb_formats_paiement", "max_transactions_jour",
    "ratio_recu_envoye",
    "nb_contreparties_envoi", "nb_contreparties_reception",
    "nb_contreparties_reciproques", "ratio_reciprocite",
]

LABEL = "label_laundering"


def prepare_features(df):
    """Nettoie les features (valeurs manquantes, infinies) et sépare X / y."""
    df = df.copy()
    df[FEATURES] = df[FEATURES].fillna(0)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], 0)

    X = df[FEATURES]
    y = df[LABEL]
    return X, y
