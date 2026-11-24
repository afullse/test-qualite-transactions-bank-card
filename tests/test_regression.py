"""Non-régression vs référence figée (reports/baseline.json, versionnée, jamais recalculée en CI)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from evaluation import evaluate_model
from prediction import predict_fraud

BASELINE = Path(__file__).resolve().parent.parent / "reports" / "baseline.json"
TOL = {"recall": 0.03, "precision": 0.05, "pr_auc": 0.03, "fpr": 0.01}


def current_metrics(trained, splits):
    model, thr = trained
    _, Xva, _, _, yva, _ = splits
    pred, proba = predict_fraud(model, Xva, thr)
    return evaluate_model(yva, pred, proba), thr


def test_T06_reproducible(splits):
    from app.training import train_model
    Xtr, Xva, _, ytr, _, _ = splits
    a = train_model(Xtr, ytr).predict_proba(Xva)[:, 1]
    b = train_model(Xtr, ytr).predict_proba(Xva)[:, 1]
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_no_regression_vs_baseline(trained, splits):
    if os.environ.get("UPDATE_BASELINE") == "1":  # action manuelle et tracée uniquement
        m, thr = current_metrics(trained, splits)
        BASELINE.write_text(json.dumps({"metrics": m, "threshold": thr, "tolerances": TOL}, indent=2))
        pytest.skip("Baseline mise à jour manuellement")
    assert BASELINE.exists(), "Baseline absente"
    base = json.loads(BASELINE.read_text())
    m, _ = current_metrics(trained, splits)
    b = base["metrics"]
    assert m["recall"] >= b["recall"] - TOL["recall"]
    assert m["precision"] >= b["precision"] - TOL["precision"]
    assert m["pr_auc"] >= b["pr_auc"] - TOL["pr_auc"]
    assert m["fpr"] <= b["fpr"] + TOL["fpr"]
