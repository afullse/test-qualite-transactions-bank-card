import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from app.evaluation import evaluate_model
from app.preprocessing import (FEATURES, REQUIRED_COLUMNS, DataValidationError,
                               preprocess_data, split_data, validate_data)


def test_standard_scaler():
    data = pd.DataFrame({"Amount": [100.0, 200.0, 300.0]})
    scaled = StandardScaler().fit_transform(data)
    assert np.isfinite(scaled).all()
    assert abs(scaled.mean()) < 1e-10
    assert np.isclose(scaled.std(), 1.0)


def test_T01_all_required_columns_present(df):
    assert validate_data(df) is df
    assert set(REQUIRED_COLUMNS) <= set(df.columns)


@pytest.mark.parametrize("col", ["Time", "V14", "Amount", "Class"])
def test_missing_column_rejected(df, col):
    with pytest.raises(DataValidationError, match="manquantes"):
        validate_data(df.drop(columns=[col]))


def test_duplicated_column_rejected(df):
    bad = pd.concat([df, df[["V1"]]], axis=1)
    with pytest.raises(DataValidationError, match="dupliquées"):
        validate_data(bad)


def test_nan_rejected(df):
    bad = df.copy(); bad.loc[bad.index[3], "V2"] = np.nan
    with pytest.raises(DataValidationError, match="manquantes"):
        validate_data(bad)


@pytest.mark.parametrize("val", [np.inf, -np.inf])
def test_inf_rejected(df, val):
    bad = df.copy(); bad.loc[bad.index[0], "V5"] = val
    with pytest.raises(DataValidationError, match="infinies"):
        validate_data(bad)


def test_empty_rejected(df):
    with pytest.raises(DataValidationError, match="vide"):
        validate_data(df.iloc[0:0])


def test_T02_wrong_type_rejected(df):
    bad = df.copy(); bad["Amount"] = bad["Amount"].astype(str)
    with pytest.raises(DataValidationError, match="Type"):
        validate_data(bad)


@pytest.mark.parametrize("target", [2, -1, 0.5])
def test_invalid_target_rejected(df, target):
    bad = df.copy(); bad["Class"] = bad["Class"].astype(float); bad.loc[bad.index[0], "Class"] = target
    with pytest.raises(DataValidationError, match="cible"):
        validate_data(bad)


@pytest.mark.parametrize("amount", [-0.01, 1e9])
def test_amount_out_of_domain_rejected(df, amount):
    bad = df.copy(); bad.loc[bad.index[0], "Amount"] = amount
    with pytest.raises(DataValidationError, match="Amount"):
        validate_data(bad)


def test_amount_boundaries_accepted(df):
    ok = df.head(2).copy(); ok["Amount"] = [0.0, 1_000_000.0]
    out = preprocess_data(ok)
    assert out["Amount"].iloc[0] == 0.0
    assert np.isfinite(out.to_numpy()).all()


def test_single_observation(df):
    out = preprocess_data(df.head(1))
    assert out.shape == (1, len(FEATURES) + 1)


def test_T03_preprocess_finite_and_shape(df):
    out = preprocess_data(df)
    assert out.shape == (len(df), len(FEATURES) + 1)   # -Time +sin +cos
    assert np.isfinite(out.to_numpy()).all()
    assert out.index.equals(df.index)


def test_preprocess_is_deterministic_and_pure(df):
    before = df.copy()
    a, b = preprocess_data(df), preprocess_data(df)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(df, before)  # pas d'effet de bord


def test_time_cyclic_encoding():
    row = {c: 0.0 for c in FEATURES}
    d = pd.DataFrame([row, {**row, "Time": 86400.0}])
    out = preprocess_data(d)
    assert np.allclose(out.iloc[0], out.iloc[1])  # même heure, jour différent


def test_split_stratified_and_disjoint(df):
    Xtr, Xva, Xte, ytr, yva, yte = split_data(df)
    assert set(Xtr.index).isdisjoint(Xva.index) and set(Xtr.index).isdisjoint(Xte.index)
    assert set(Xva.index).isdisjoint(Xte.index)
    assert len(Xtr) + len(Xva) + len(Xte) == len(df)
    r = df["Class"].mean()
    for y in (ytr, yva, yte):
        assert abs(y.mean() - r) < 0.005


def test_evaluate_model_known_values():
    y = [0, 0, 0, 1, 1]; p = [0, 1, 0, 1, 0]; pr = [.1, .8, .2, .9, .4]
    m = evaluate_model(y, p, pr)
    assert (m["TP"], m["TN"], m["FP"], m["FN"]) == (1, 2, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert np.isclose(m["fpr"], 1 / 3) and m["fnr"] == 0.5


def test_accuracy_paradox():
    """Un modèle qui ne prédit jamais la fraude a une accuracy élevée et un rappel nul."""
    y = np.array([0] * 998 + [1] * 2)
    m = evaluate_model(y, np.zeros(1000, int), np.zeros(1000))
    assert m["accuracy"] == 0.998 and m["recall"] == 0.0


def test_evaluate_rejects_inconsistent_sizes():
    with pytest.raises(ValueError):
        evaluate_model([0, 1], [0], [0.1, 0.2])
