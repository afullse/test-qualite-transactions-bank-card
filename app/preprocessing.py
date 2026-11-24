"""Contrat de données, validation et préparation des variables."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"
REQUIRED_COLUMNS = FEATURES + [TARGET]
AMOUNT_MAX = 1_000_000.0  # borne métier documentée (au-delà : rejet)


class DataValidationError(ValueError):
    """Levée lorsqu'une donnée viole le contrat (rejet explicite)."""


def validate_data(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Contrôle le schéma, les types et les valeurs. Rejette toute violation."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("L'entrée doit être un DataFrame pandas.")
    if df.empty:
        raise DataValidationError("Jeu de données vide.")
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        raise DataValidationError(f"Colonnes dupliquées : {dups}")
    required = REQUIRED_COLUMNS if require_target else FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Colonnes manquantes : {missing}")
    for c in FEATURES:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            raise DataValidationError(f"Type non numérique pour la colonne {c}")
    values = df[FEATURES].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise DataValidationError("Valeurs manquantes détectées.")
    if np.isinf(values).any():
        raise DataValidationError("Valeurs infinies détectées.")
    if (df["Amount"] < 0).any() or (df["Amount"] > AMOUNT_MAX).any():
        raise DataValidationError("Amount hors du domaine [0, AMOUNT_MAX].")
    if (df["Time"] < 0).any():
        raise DataValidationError("Time négatif.")
    if require_target:
        if not set(pd.unique(df[TARGET])).issubset({0, 1}):
            raise DataValidationError("La cible doit valoir 0 ou 1.")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transformation déterministe sans état (aucun apprentissage => pas de fuite).

    - Amount -> log1p(Amount) (réduit l'asymétrie)
    - Time   -> heure de la journée (cyclique sin/cos) au lieu d'un compteur absolu
    La standardisation (apprise) est faite dans le Pipeline, ajustée sur train.
    """
    validate_data(df, require_target=False)
    out = df[FEATURES].astype(float).copy()
    out["Amount"] = np.log1p(out["Amount"])
    hours = (out["Time"] % 86400) / 3600.0
    out["Hour_sin"] = np.sin(2 * np.pi * hours / 24)
    out["Hour_cos"] = np.cos(2 * np.pi * hours / 24)
    out = out.drop(columns=["Time"])
    if not np.isfinite(out.to_numpy()).all():
        raise DataValidationError("Valeurs non finies après prétraitement.")
    return out


def split_data(df, seed=42, val_size=0.2, test_size=0.2):
    """Partition stratifiée train/validation/test AVANT tout apprentissage."""
    from sklearn.model_selection import train_test_split

    X, y = df[FEATURES], df[TARGET].astype(int)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    rel = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel, stratify=y_tmp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test
