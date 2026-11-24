"""Exécution complète : python -m app.run_experiment [--data data/creditcard.csv]"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay  # noqa: E402

from evaluation import evaluate_model, expected_cost  # noqa: E402
from prediction import predict_fraud  # noqa: E402
from preprocessing import split_data, validate_data  # noqa: E402
from synthetic import make_synthetic  # noqa: E402
from training import load_model, save_model, select_threshold, train_model  # noqa: E402

R = Path("reports")


def main(data=None, min_recall=0.80):
    R.mkdir(exist_ok=True)
    df = pd.read_csv(data) if data else make_synthetic(20000, seed=1)
    validate_data(df)
    Xtr, Xva, Xte, ytr, yva, yte = split_data(df)
    results, fig1, ax1 = {}, *plt.subplots()
    fig2, ax2 = plt.subplots()
    for name in ["logreg", "tree"]:
        pipe = train_model(Xtr, ytr, name)
        thr = select_threshold(yva, pipe.predict_proba(Xva)[:, 1], min_recall)
        path = save_model(pipe, thr, f"models/{name}.joblib")
        pipe2, thr2 = load_model(path)
        res = {}
        for split, X, y in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
            p, pr = predict_fraud(pipe2, X, thr2)
            res[split] = evaluate_model(y, p, pr)
            res[split]["cost"] = expected_cost(res[split])
        res["threshold"] = thr
        # latence
        one = Xte.iloc[[0]]
        lat = []
        for _ in range(300):
            t = time.perf_counter(); pipe2.predict_proba(one); lat.append(time.perf_counter() - t)
        t = time.perf_counter(); pipe2.predict_proba(Xte); dt = time.perf_counter() - t
        res["perf"] = {"p50_ms": float(np.percentile(lat, 50) * 1e3),
                       "p95_ms": float(np.percentile(lat, 95) * 1e3),
                       "throughput_per_s": float(len(Xte) / dt),
                       "model_kb": path.stat().st_size / 1024}
        results[name] = res
        pr_te = pipe2.predict_proba(Xte)[:, 1]
        RocCurveDisplay.from_predictions(yte, pr_te, name=name, ax=ax1)
        PrecisionRecallDisplay.from_predictions(yte, pr_te, name=name, ax=ax2)
    fig1.savefig(R / "roc.png", dpi=120); fig2.savefig(R / "pr.png", dpi=120)
    (R / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v["test"] | {"thr": v["threshold"]} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--data") 
    a.add_argument("--min-recall", type=float, default=0.80)
    ns = a.parse_args()
    main(ns.data, ns.min_recall)
