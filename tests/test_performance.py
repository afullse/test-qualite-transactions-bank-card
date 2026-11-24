import os
import time

import numpy as np
import psutil


def test_latency_and_memory(model, X_test, tmp_path):
    one = X_test.iloc[[0]]
    for _ in range(20):
        model.predict_proba(one)  # warm-up
    lat = []
    for _ in range(200):
        t = time.perf_counter(); model.predict_proba(one); lat.append(time.perf_counter() - t)
    p95_ms = np.percentile(lat, 95) * 1e3
    assert p95_ms < 50, f"P95={p95_ms:.2f} ms"   # budget hors HTTP (cf. plan de test)
    t = time.perf_counter(); model.predict_proba(X_test); dt = time.perf_counter() - t
    assert len(X_test) / dt > 1000
    rss = psutil.Process(os.getpid()).memory_info().rss / 2**20
    assert rss < 2048


def test_no_secrets_in_repo():
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    pat = re.compile(r"(AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|-----BEGIN (RSA )?PRIVATE KEY|password\s*=\s*['\"][^'\"]+)")
    for f in root.rglob("*"):
        if f.suffix in {".py", ".yml", ".yaml", ".txt", ".md", ".json", ".cfg", ".toml"} and ".git" not in f.parts:
            assert not pat.search(f.read_text(errors="ignore")), f"Secret potentiel : {f}"
