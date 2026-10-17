"""Навчання класифікатора атак з перехресною перевіркою leave-one-subject-out.

Порівнюються базовий поріг на мінімумі s_ref (ThresholdClassifier) і
логістична регресія на 26 ознаках (LogisticClassifier). Для кожного учасника
модель навчається на решті учасників; поріг рішення τ підбирається на
внутрішній перехресній перевірці навчальних учасників так, щоб BPCER не
перевищував target_bpcer. Показники APCER, BPCER, ACER, BPCER20, EER і AUC
обчислюються за ДСТУ ISO/IEC 30107-3:2023 на рівні сеансу (оцінка сеансу –
максимум оцінок викликів) і довідково на рівні викликів.

Приклади:
    python scripts/train.py --synthetic                  # перевірка на синтетичних даних
    python scripts/train.py --data data/features.csv     # пілотний набір
    python scripts/train.py --data data/features.csv --install   # скопіювати модель у models/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from pad.classifiers import LogisticClassifier, ThresholdClassifier  # noqa: E402
from pad.cli import setup_console  # noqa: E402
from pad.dataset import read_csv, to_arrays, write_csv  # noqa: E402
from pad.metrics import evaluate, rates_from_decisions, threshold_for_bpcer  # noqa: E402
from pad.synthetic import make_synthetic_dataset  # noqa: E402

MODELS = {
    "threshold": lambda cfg: ThresholdClassifier(cfg.get("threshold_feature", "s_ref_during_min")),
    "logistic": lambda cfg: LogisticClassifier(C=cfg.get("C", 1.0), seed=cfg.get("seed", 0)),
}


def session_scores(challenge_scores: np.ndarray, sessions: np.ndarray) -> dict[str, float]:
    """Оцінка сеансу – максимум оцінок його викликів."""
    out: dict[str, float] = {}
    for s, p in zip(sessions, challenge_scores):
        out[s] = max(out.get(s, -np.inf), float(p))
    return out


def session_table(data: dict) -> dict[str, dict]:
    """Мітка, учасник і вид атаки для кожного сеансу."""
    table = {}
    for s, y, subj, sp in zip(data["session"], data["y"], data["subject"], data["species"]):
        table[s] = {"label": int(y), "subject": subj, "species": sp or "attack"}
    return table


def oof_threshold(make, data: dict, idx: np.ndarray, target: float) -> float:
    """Поріг τ за внутрішньою LOSO-перевіркою на учасниках idx (BPCER <= target)."""
    subjects = np.unique(data["subject"][idx])
    scores = np.full(len(data["y"]), np.nan)
    if len(subjects) < 2:
        model = make().fit(data["X"][idx], data["y"][idx])
        scores[idx] = model.predict_batch(data["X"][idx])
    else:
        for subj in subjects:
            tr = idx[data["subject"][idx] != subj]
            te = idx[data["subject"][idx] == subj]
            if len(np.unique(data["y"][tr])) < 2:
                continue
            scores[te] = make().fit(data["X"][tr], data["y"][tr]).predict_batch(data["X"][te])
    ok = idx[np.isfinite(scores[idx])]
    sess = session_scores(scores[ok], data["session"][ok])
    table = session_table(data)
    keys = list(sess)
    return threshold_for_bpcer([sess[k] for k in keys], [table[k]["label"] for k in keys], target)


def loso(data: dict, name: str, cfg: dict) -> dict:
    """Зовнішня LOSO-перевірка однієї моделі."""
    make = lambda: MODELS[name](cfg)  # noqa: E731
    target = cfg.get("target_bpcer", 0.05)
    table = session_table(data)
    ch_scores = np.full(len(data["y"]), np.nan)
    sess_rows, folds = [], []
    for subj in np.unique(data["subject"]):
        tr = np.nonzero(data["subject"] != subj)[0]
        te = np.nonzero(data["subject"] == subj)[0]
        if len(np.unique(data["y"][tr])) < 2:
            continue
        tau = oof_threshold(make, data, tr, target)
        model = make().fit(data["X"][tr], data["y"][tr])
        ch_scores[te] = model.predict_batch(data["X"][te])
        sess = session_scores(ch_scores[te], data["session"][te])
        fold_rows = []
        for s, score in sess.items():
            row = {"subject": subj, "session_uid": s, "label": table[s]["label"],
                   "species": table[s]["species"], "score": score, "threshold": tau,
                   "is_attack": int(score >= tau)}
            fold_rows.append(row)
        sess_rows.extend(fold_rows)
        r = rates_from_decisions([r["is_attack"] for r in fold_rows], [r["label"] for r in fold_rows],
                                 [r["species"] for r in fold_rows])
        folds.append({"model": name, "test_subject": subj, "threshold": tau,
                      "n_sessions": len(fold_rows), **{k: r[k] for k in ("apcer", "bpcer", "acer")}})

    labels = np.array([r["label"] for r in sess_rows])
    scores = np.array([r["score"] for r in sess_rows])
    species = [r["species"] for r in sess_rows]
    pooled = evaluate(scores, labels, threshold=0.5, species=species)
    op = rates_from_decisions([r["is_attack"] for r in sess_rows], labels, species)
    session_metrics = {
        "operating_point": {**op, "note": f"порогу τ кожного фолду відповідає BPCER <= {target} на валідації"},
        "bpcer20": pooled["bpcer20"], "eer": pooled["eer"], "auc": pooled["auc"],
        "n_bona_fide": pooled["n_bona_fide"], "n_attack": pooled["n_attack"],
    }
    ok = np.isfinite(ch_scores)
    challenge = evaluate(ch_scores[ok], data["y"][ok], threshold=0.5, species=data["species"][ok])
    per_type = {}
    for kind in np.unique(data["challenge_type"]):
        sel = ok & (data["challenge_type"] == kind)
        e = evaluate(ch_scores[sel], data["y"][sel], threshold=0.5, species=data["species"][sel])
        per_type[str(kind)] = {"eer": e["eer"], "auc": e["auc"], "n": int(sel.sum())}
    return {"session": session_metrics, "challenge": {k: challenge[k] for k in ("eer", "auc", "bpcer20")},
            "per_challenge_type": per_type, "folds": folds, "session_rows": sess_rows,
            "challenge_scores": ch_scores}


def plot_curves(results: dict, out: Path) -> None:
    """DET- і ROC-криві на рівні сеансу для всіх моделей (PNG)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from sklearn.metrics import det_curve, roc_curve

    fig_det, ax_det = plt.subplots(figsize=(5.5, 5))
    fig_roc, ax_roc = plt.subplots(figsize=(5.5, 5))
    for name, res in results.items():
        y = [r["label"] for r in res["session_rows"]]
        s = [r["score"] for r in res["session_rows"]]
        if len(set(y)) < 2:
            continue
        bpcer, apcer, _ = det_curve(y, s)
        eps = 1e-3
        ax_det.plot(norm.ppf(np.clip(bpcer, eps, 1 - eps)), norm.ppf(np.clip(apcer, eps, 1 - eps)),
                    label=f"{name} (EER {res['session']['eer']:.3f})".replace(".", ","))
        fpr, tpr, _ = roc_curve(y, s)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC {res['session']['auc']:.3f})".replace(".", ","))
    ticks = np.array([0.001, 0.01, 0.05, 0.2, 0.5, 0.8])
    ax_det.set_xticks(norm.ppf(ticks), [f"{t:g}".replace(".", ",") for t in ticks])
    ax_det.set_yticks(norm.ppf(ticks), [f"{t:g}".replace(".", ",") for t in ticks])
    ax_det.set_xlabel("BPCER")
    ax_det.set_ylabel("APCER")
    ax_det.set_title("DET-крива (рівень сеансу)")
    ax_det.grid(alpha=0.3)
    ax_det.legend()
    ax_roc.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    ax_roc.set_xlabel("BPCER")
    ax_roc.set_ylabel("1 – APCER")
    ax_roc.set_title("ROC-крива (рівень сеансу)")
    ax_roc.grid(alpha=0.3)
    ax_roc.legend()
    fig_det.tight_layout()
    fig_roc.tight_layout()
    fig_det.savefig(out / "det.png", dpi=150)
    fig_roc.savefig(out / "roc.png", dpi=150)
    plt.close("all")


def split_hash(subjects) -> str:
    """Хеш розбиття LOSO (відтворюваність, FR-67)."""
    return hashlib.sha256(",".join(sorted(map(str, set(subjects)))).encode()).hexdigest()[:16]


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="CSV ознак (scripts/build_dataset.py)")
    src.add_argument("--synthetic", action="store_true", help="синтетичні дані")
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment_pilot.yaml"))
    ap.add_argument("--out", default=None, help="тека результатів (за замовчуванням runs/train_<час>)")
    ap.add_argument("--install", action="store_true", help="скопіювати модель у models/pad_logistic.json")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    out = Path(args.out) if args.out else ROOT / "runs" / f"train_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        syn = cfg.get("synthetic", {})
        rows = make_synthetic_dataset(syn.get("n_subjects", 6), syn.get("sessions_per_class", 8), seed)
        write_csv(rows, out / "features_synthetic.csv")
        source = "synthetic"
    else:
        rows = read_csv(args.data)
        source = str(args.data)
    data = to_arrays(rows)
    n_subj = len(np.unique(data["subject"]))
    if n_subj < 2 or len(np.unique(data["y"])) < 2:
        sys.exit("потрібно щонайменше 2 учасники і обидва класи (справжні подання та атаки)")
    print(f"дані: {source}; викликів {len(rows)}, сеансів {len(np.unique(data['session']))}, "
          f"учасників {n_subj}")

    model_cfg = {**cfg.get("model", {}), "seed": seed, "target_bpcer": cfg.get("target_bpcer", 0.05)}
    results = {name: loso(data, name, model_cfg) for name in cfg.get("models", ["threshold", "logistic"])}

    # підсумкові моделі на всіх даних; рекомендований поріг – за LOSO-оцінками сеансів
    final = {}
    for name in results:
        model = MODELS[name](model_cfg).fit(data["X"], data["y"])
        rows_s = results[name]["session_rows"]
        tau = threshold_for_bpcer([r["score"] for r in rows_s], [r["label"] for r in rows_s],
                                  model_cfg["target_bpcer"])
        d = model.to_dict()
        d["recommended_threshold"] = tau
        d["trained_on"] = source
        path = out / f"model_{name}.json"
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        final[name] = {"path": str(path), "recommended_threshold": tau}
        if name == "logistic":
            try:
                model.export_onnx(out / "model_logistic.onnx")
                final[name]["onnx"] = str(out / "model_logistic.onnx")
            except ImportError:
                print("пакет onnx не встановлено, експорт у ONNX пропущено")
            final[name]["top_features"] = model.importance()[:8]

    summary = {
        "source": source, "seed": seed, "config": cfg, "split": "leave-one-subject-out",
        "split_hash": split_hash(data["subject"]), "subjects": sorted(map(str, np.unique(data["subject"]))),
        "metrics": {name: {k: r[k] for k in ("session", "challenge", "per_challenge_type")}
                    for name, r in results.items()},
        "final_models": final,
    }
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")
    with open(out / "folds.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "test_subject", "threshold", "n_sessions",
                                           "apcer", "bpcer", "acer"])
        w.writeheader()
        for r in results.values():
            w.writerows(r["folds"])
    with open(out / "session_scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "subject", "session_uid", "label", "species",
                                           "score", "threshold", "is_attack"])
        w.writeheader()
        for name, r in results.items():
            w.writerows({"model": name, **row} for row in r["session_rows"])
    plot_curves(results, out)
    shutil.copy(args.config, out / "config.yaml")

    print(f"\n{'модель':<10} {'APCER':>7} {'BPCER':>7} {'ACER':>7} {'BPCER20':>8} {'EER':>7} {'AUC':>7}")
    for name, r in results.items():
        s, op = r["session"], r["session"]["operating_point"]
        print(f"{name:<10} {op['apcer']:7.3f} {op['bpcer']:7.3f} {op['acer']:7.3f} "
              f"{s['bpcer20']:8.3f} {s['eer']:7.3f} {s['auc']:7.3f}")
    print(f"\nрезультати: {out}")
    if args.install:
        dst = ROOT / "models" / "pad_logistic.json"
        shutil.copy(out / "model_logistic.json", dst)
        print(f"модель скопійовано в {dst}; поріг у configs/pad.yaml: "
              f"{final['logistic']['recommended_threshold']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
