import numpy as np
import pytest

from prediction import predict_fraud
from training import select_threshold, train_model


def test_T04_probabilities_valid(model, X_test):
    p = model.predict_proba(X_test)[:, 1]
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()


def test_predict_fraud_contract(model, X_test):
    pred, proba = predict_fraud(model, X_test, 0.5)
    assert pred.shape == (len(X_test),)
    assert set(np.unique(pred)) <= {0, 1}
    assert np.array_equal(pred, (proba >= 0.5).astype(int))


def test_threshold_monotonic(model, X_test):
    n = [predict_fraud(model, X_test, t)[0].sum() for t in (0.1, 0.5, 0.9)]
    assert n[0] >= n[1] >= n[2]


@pytest.mark.parametrize("t", [-0.1, 1.5])
def test_invalid_threshold(model, X_test, t):
    with pytest.raises(ValueError):
        predict_fraud(model, X_test, t)


def test_extra_field_rejected(model, X_test):
    bad = X_test.copy(); bad["card_number"] = "4970..."
    with pytest.raises(ValueError, match="non attendus"):
        predict_fraud(model, bad)


def test_train_requires_two_classes(splits):
    Xtr, _, _, ytr, _, _ = splits
    with pytest.raises(ValueError):
        train_model(Xtr[ytr == 0], ytr[ytr == 0])


def test_unknown_model():
    with pytest.raises(ValueError):
        from app.training import build_pipeline
        build_pipeline("xgb")


def test_select_threshold_meets_recall(trained, splits):
    model, _ = trained
    _, Xva, _, _, yva, _ = splits
    pr = model.predict_proba(Xva)[:, 1]
    thr = select_threshold(yva, pr, 0.8)
    assert ((pr >= thr)[yva.to_numpy() == 1]).mean() >= 0.8


def test_logs_contain_no_transaction_data(model, X_test, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="ai_testguard")
    predict_fraud(model, X_test.head(5))
    txt = caplog.text
    assert "Amount" not in txt and str(X_test.iloc[0]["V1"])[:6] not in txt
