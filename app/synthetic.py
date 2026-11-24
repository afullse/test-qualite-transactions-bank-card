"""Générateur de données synthétiques au schéma Kaggle (tests/CI sans le vrai fichier)."""
import numpy as np
import pandas as pd

from app.preprocessing import FEATURES


def make_synthetic(n: int = 4000, fraud_rate: float = 0.02, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    y[:10] = 1
    X = rng.normal(size=(n, 28))
    X[y == 1, :5] += rng.normal(2.0, 1.0, size=((y == 1).sum(), 5))
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = np.sort(rng.uniform(0, 172800, n))
    df["Amount"] = np.round(rng.lognormal(3.5, 1.2, n), 2)
    df["Class"] = y
    return df[FEATURES + ["Class"]]
