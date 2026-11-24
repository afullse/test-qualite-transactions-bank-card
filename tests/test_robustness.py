"""Robustesse : mesure de sensibilité (pas d'invariance imposée)."""
import json
from pathlib import Path

import numpy as np
import pytest

from app.evaluation import evaluate_model
from app.prediction import predict_fraud
from app.preprocessing import FEATURES, DataValidationError

V = [f"V{i}" for i in range(1, 29)]


@pytest.mark.parametrize("cols", [["V1"], ["V1", "Amount"]])
def test_missing_values_rejected(model, X_test, cols):
    bad = X_test.copy(); bad.loc[bad.index[:3], cols] = np.nan
    with pytest.raises(DataValidationError):
        predict_fraud(model, bad)


def test_non_numeric_rejected(model, X_test):
    bad = X_test.copy().astype({"V3": object}); bad.iloc[0, bad.columns.get_loc("V3")] = "abc"
    with pytest.raises(DataValidationError):
        predict_fraud(model, bad)


def test_extreme_amount_within_domain_processed(model, X_test):
    x = X_test.head(20).copy(); x["Amount"] = 999_999.0
    _, p = predict_fraud(model, x)
    assert np.isfinite(p).all()


def test_amount_beyond_domain_rejected(model, X_test):
    x = X_test.head(5).copy(); x["Amount"] = 1e12
    with pytest.raises(DataValidationError):
        predict_fraud(model, x)


def test_invalid_schema_rejected(model, X_test):
    with pytest.raises(DataValidationError):
        predict_fraud(model, X_test.drop(columns=["V10"]))


def sensitivity(model, X, y, thr, eps, seed=0):
    """Bruit gaussien proportionnel à l'écart-type de chaque V_i (sigma*eps),
    Amount : variation relative +/- eps (arrondi au centime) ; Time inchangé."""
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    Xp[V] = Xp[V] + rng.normal(0, 1, Xp[V].shape) * X[V].std().to_numpy() * eps
    Xp["Amount"] = np.round(Xp["Amount"] * (1 + rng.uniform(-eps, eps, len(X))), 2).clip(0)
    p0, pr0 = predict_fraud(model, X, thr)
    p1, pr1 = predict_fraud(model, Xp, thr)
    m0, m1 = evaluate_model(y, p0, pr0), evaluate_model(y, p1, pr1)
    return {"eps": eps, "mean_abs_dproba": float(np.abs(pr1 - pr0).mean()),
            "max_abs_dproba": float(np.abs(pr1 - pr0).max()),
            "class_flip_rate": float((p0 != p1).mean()),
            "d_recall": m1["recall"] - m0["recall"], "d_precision": m1["precision"] - m0["precision"],
            "d_fpr": m1["fpr"] - m0["fpr"], "invalid": 0}


def test_sensitivity_small_perturbation(trained, splits):
    model, thr = trained
    _, _, Xte, _, _, yte = splits
    rows = [sensitivity(model, Xte, yte, thr, e) for e in (0.01, 0.05, 0.10)]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/robustness.json").write_text(json.dumps(rows, indent=2))
    small = rows[0]
    assert small["class_flip_rate"] <= 0.01      # tolérance définie au plan de test
    assert abs(small["d_recall"]) <= 0.05
    assert rows[0]["mean_abs_dproba"] <= rows[2]["mean_abs_dproba"]  # sensibilité croissante


def test_distribution_shift_detected(splits):
    """Dérive : un décalage de V14 doit être détecté par un test KS (alerte)."""
    from scipy.stats import ks_2samp
    Xtr, _, Xte, *_ = splits
    shifted = Xte.copy(); shifted["V14"] += 1.0
    assert ks_2samp(Xtr["V14"], Xte["V14"]).pvalue > 0.01       # pas d'alerte
    assert ks_2samp(Xtr["V14"], shifted["V14"]).pvalue < 0.01   # alerte


try:
    from hypothesis import given, settings, strategies as st

    @settings(max_examples=50, deadline=None)
    @given(st.lists(st.floats(-50, 50), min_size=28, max_size=28),
           st.floats(0, 25_000), st.floats(0, 172_800))
    def test_property_valid_input_gives_valid_proba(trained, vs, amount, time):
        import pandas as pd
        model, thr = trained
        row = dict(zip(V, vs)); row.update(Amount=amount, Time=time)
        _, p = predict_fraud(model, pd.DataFrame([row])[FEATURES], thr)
        assert 0 <= p[0] <= 1
except ImportError:  # Hypothesis optionnel en local
    pass
