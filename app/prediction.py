"""Production des prédictions avec contrat d'entrée/sortie."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from preprocessing import FEATURES, validate_data

logger = logging.getLogger("ai_testguard")


def predict_fraud(model, X: pd.DataFrame, threshold: float = 0.5):
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Le seuil doit être dans [0, 1].")
    extra = [c for c in X.columns if c not in FEATURES + ["Class"]]
    if extra:
        raise ValueError(f"Champs non attendus : {extra}")
    validate_data(X, require_target=False)
    proba = model.predict_proba(X[FEATURES])[:, 1]
    if not (np.isfinite(proba).all() and (proba >= 0).all() and (proba <= 1).all()):
        raise RuntimeError("Probabilités hors contrat.")
    pred = (proba >= threshold).astype(int)
    # journalisation agrégée uniquement : aucune donnée de transaction dans les logs
    logger.info("Prédictions: n=%d, signalées=%d", len(pred), int(pred.sum()))
    return pred, proba
