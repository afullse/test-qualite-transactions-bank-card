"""Calcul des métriques adaptées aux classes déséquilibrées."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def evaluate_model(y_true, y_pred, y_proba) -> dict:
    y_true, y_pred, y_proba = map(np.asarray, (y_true, y_pred, y_proba))
    if not (len(y_true) == len(y_pred) == len(y_proba)):
        raise ValueError("Tailles incohérentes.")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_proba)) if both else float("nan"),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }


def expected_cost(m: dict, cost_fp: float = 5.0, cost_fn: float = 120.0) -> float:
    """Coût métier hypothétique (à valider avec la banque)."""
    return m["FP"] * cost_fp + m["FN"] * cost_fn
