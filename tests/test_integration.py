import numpy as np
import pytest

from app.evaluation import evaluate_model
from app.prediction import predict_fraud
from app.preprocessing import DataValidationError
from app.training import load_model, save_model


def test_end_to_end_save_reload(trained, splits, tmp_path):
    model, thr = trained
    _, _, Xte, _, _, yte = splits
    p1 = model.predict_proba(Xte)[:, 1]
    path = save_model(model, thr, tmp_path / "m.joblib")
    assert path.with_suffix(".json").exists()
    m2, thr2 = load_model(path)
    assert thr2 == thr
    pred, p2 = predict_fraud(m2, Xte, thr2)
    np.testing.assert_allclose(p1, p2, atol=1e-12)
    m = evaluate_model(yte, pred, p2)
    assert m["TP"] + m["TN"] + m["FP"] + m["FN"] == len(Xte)


def test_scaler_fitted_on_train_only(trained, splits):
    """Fuite : les moyennes du scaler doivent correspondre au train, pas au test."""
    model, _ = trained
    Xtr, _, Xte, *_ = splits
    prep = model.named_steps["prep"]
    sc = model.named_steps["scaler"]
    assert sc.n_samples_seen_ == len(Xtr)
    np.testing.assert_allclose(sc.mean_, prep.transform(Xtr).mean().to_numpy(), atol=1e-9)
    assert not np.allclose(sc.mean_, prep.transform(Xte).mean().to_numpy(), atol=1e-9)


def test_T05_test_set_not_used_for_threshold(trained, splits):
    """Le seuil est recalculable à l'identique sans le jeu de test."""
    from app.training import select_threshold
    model, thr = trained
    _, Xva, _, _, yva, _ = splits
    assert select_threshold(yva, model.predict_proba(Xva)[:, 1]) == thr


def test_dimension_consistency(model, X_test):
    out = model[:-1].transform(X_test)
    assert out.shape == (len(X_test), model.named_steps["scaler"].n_features_in_)
    assert np.isfinite(out).all()


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "absent.joblib")


def test_load_corrupted_file(tmp_path):
    import joblib
    p = tmp_path / "bad.joblib"; joblib.dump([1, 2], p)
    with pytest.raises(ValueError):
        load_model(p)


def test_invalid_input_raises_explicit_error(model, X_test):
    bad = X_test.copy(); bad.iloc[0, 0] = np.nan
    with pytest.raises(DataValidationError):
        predict_fraud(model, bad)
