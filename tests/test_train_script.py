"""Перевірка скрипта навчання на синтетичних даних."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_train_synthetic(tmp_path):
    out = tmp_path / "run"
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--synthetic", "--out", str(out)],
                         capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert res.returncode == 0, res.stderr
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["split"] == "leave-one-subject-out"
    for name in ("threshold", "logistic"):
        s = metrics["metrics"][name]["session"]
        assert 0.0 <= s["eer"] <= 1.0 and 0.0 <= s["auc"] <= 1.0
        assert {"apcer", "bpcer", "acer"} <= set(s["operating_point"])
        assert "bpcer20" in s
    for f in ("det.png", "roc.png", "folds.csv", "session_scores.csv", "model_logistic.json"):
        assert (out / f).exists(), f
