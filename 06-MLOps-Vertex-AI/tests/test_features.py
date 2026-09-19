import numpy as np
import pandas as pd

from aml_detection.features import FEATURES, LABEL, prepare_features


def test_prepare_features_fills_na_and_inf():
    df = pd.DataFrame({f: [1.0, np.nan, np.inf, -np.inf] for f in FEATURES})
    df[LABEL] = [0, 1, 0, 1]

    X, y = prepare_features(df)

    assert list(X.columns) == FEATURES
    assert not X.isna().any().any()
    assert np.isfinite(X.to_numpy()).all()
    assert list(y) == [0, 1, 0, 1]


def test_prepare_features_does_not_mutate_input():
    df = pd.DataFrame({f: [1.0, np.nan] for f in FEATURES})
    df[LABEL] = [0, 1]
    original_na_count = df.isna().sum().sum()

    prepare_features(df)

    assert df.isna().sum().sum() == original_na_count
