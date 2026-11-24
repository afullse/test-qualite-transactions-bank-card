"""Entraînement, choix du seuil (validation uniquement), sauvegarde."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from preprocessing import preprocess_data

SEED = 42


def build_pipeline(model_name: str = "logreg", seed: int = SEED) -> Pipeline:
    if model_name == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    elif model_name == "tree":
        clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                     class_weight="balanced", random_state=seed)
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")
    return Pipeline([
        ("prep", FunctionTransformer(preprocess_data, validate=False)),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def train_model(X, y, model_name: str = "logreg", seed: int = SEED) -> Pipeline:
    if len(X) != len(y):
        raise ValueError("X et y de tailles différentes.")
    if len(np.unique(y)) < 2:
        raise ValueError("Les deux classes doivent être présentes à l'entraînement.")
    pipe = build_pipeline(model_name, seed)
    pipe.fit(X, y)
    return pipe


def select_threshold(y_val, proba_val, min_recall: float = 0.80) -> float:
    """Seuil maximisant la précision sous contrainte de rappel >= min_recall.
    Calculé EXCLUSIVEMENT sur la validation."""
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    prec, rec = prec[:-1], rec[:-1]
    ok = rec >= min_recall
    if not ok.any():
        return float(thr[np.argmax(rec)])
    idx = np.where(ok)[0][np.argmax(prec[ok])]
    return float(thr[idx])


def save_model(pipe, threshold: float, path: str | Path, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "threshold": threshold}, path)
    meta = {"threshold": threshold, "sklearn": sklearn.__version__,
            "numpy": np.__version__, "python": platform.python_version(),
            "seed": SEED, "params": {k: str(v) for k, v in pipe.named_steps["clf"].get_params().items()}}
    meta.update(extra or {})
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def load_model(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "pipeline" not in obj or "threshold" not in obj:
        raise ValueError("Fichier modèle invalide.")
    return obj["pipeline"], obj["threshold"]
