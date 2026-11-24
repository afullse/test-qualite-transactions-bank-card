import pytest

from app.preprocessing import split_data
from app.synthetic import make_synthetic
from app.training import select_threshold, train_model


@pytest.fixture(scope="session")
def df():
    return make_synthetic(4000, seed=0)


@pytest.fixture(scope="session")
def splits(df):
    return split_data(df, seed=42)


@pytest.fixture(scope="session")
def trained(splits):
    Xtr, Xva, Xte, ytr, yva, yte = splits
    model = train_model(Xtr, ytr, "logreg")
    thr = select_threshold(yva, model.predict_proba(Xva)[:, 1])
    return model, thr


@pytest.fixture
def model(trained):
    return trained[0]


@pytest.fixture
def X_test(splits):
    return splits[2]
