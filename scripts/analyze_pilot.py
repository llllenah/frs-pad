"""Звіт за пілотним набором: таблиці показників і рисунки для розділу результатів ПЗ.

Читає результати scripts/train.py (runs/pilot_*) і зашифровані записи (ключ у
PAD_DATASET_KEY) та записує в runs/pilot_report/:
    summary.json            – усі числа, з яких складено таблиці ПЗ;
    fig_series.png          – s_ref(t) і s_enr(t) справжнього подання й атаки на тих самих рухах;
    fig_det.png             – DET-криві варіантів моделі (рівень сеансу);
    fig_challenges.png      – частка виконаних викликів за типами.

    python scripts/analyze_pilot.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from pad.cli import setup_console  # noqa: E402
from pad.dataset import read_csv, to_arrays  # noqa: E402
from pad.features import FEATURE_NAMES  # noqa: E402
from pad.storage import load_key, read_json_encrypted  # noqa: E402

RUNS = {  # назва варіанта -> тека train.py
    "усі 26 ознак, max": "pilot_all",
    "усі 26 ознак, mean": "pilot_all_mean",
    "лише s_ref (13), max": "pilot_sref",
    "лише s_ref (13), mean": "pilot_sref_mean",
}
TYPES = ["cover_left", "cover_right", "cover_mouth", "turn_head", "hand_swipe"]
TYPE_UA = {"cover_left": "ліва половина", "cover_right": "права половина", "cover_mouth": "рот",
           "turn_head": "поворот голови", "hand_swipe": "рука перед обличчям"}
TYPE_UA2 = {"cover_left": "закрити ліву\nполовину", "cover_right": "закрити праву\nполовину",
            "cover_mouth": "закрити рот", "turn_head": "повернути\nголову", "hand_swipe": "провести\nрукою"}

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "axes.titlesize": 12})
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "–"))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % довірчий інтервал Вілсона для частки k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def sessions(run: str, model: str) -> list[dict]:
    with open(ROOT / "runs" / run / "session_scores.csv", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["model"] == model]


def summary_table() -> dict:
    out = {}
    for name, run in RUNS.items():
        metrics = json.loads((ROOT / "runs" / run / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        for model in ("logistic", "threshold"):
            if model == "threshold" and "mean" in name:
                continue
            rows = sessions(run, model)
            atk = [r for r in rows if r["label"] == "1"]
            bf = [r for r in rows if r["label"] == "0"]
            k_a = sum(r["is_attack"] == "0" for r in atk)     # пропущені атаки
            k_b = sum(r["is_attack"] == "1" for r in bf)      # відхилені справжні
            m = metrics[model]["session"]
            out[f"{model}: {name}"] = {
                "apcer": k_a / len(atk), "apcer_ci": wilson(k_a, len(atk)),
                "bpcer": k_b / len(bf), "bpcer_ci": wilson(k_b, len(bf)),
                "bpcer20": m["bpcer20"], "eer": m["eer"], "auc": m["auc"],
                "n_attack": len(atk), "n_bona_fide": len(bf)}
    return out


def per_subject() -> dict:
    out = {}
    for run in ("pilot_all", "pilot_sref"):
        with open(ROOT / "runs" / run / "folds.csv", encoding="utf-8") as fh:
            out[run] = [r for r in csv.DictReader(fh) if r["model"] == "logistic"]
    return out


def univariate_auc() -> list[tuple[str, float, float, float]]:
    d = to_arrays(read_csv(ROOT / "data" / "features.csv"))
    res = []
    for i, n in enumerate(FEATURE_NAMES):
        x = d["X"][:, i]
        ok = np.isfinite(x)
        a = roc_auc_score(d["y"][ok], x[ok])
        res.append((n, max(a, 1 - a), float(np.nanmean(x[d["y"] == 0])), float(np.nanmean(x[d["y"] == 1]))))
    return sorted(res, key=lambda r: -r[1])


ABLATION = {  # назва -> префікси ознак
    "усі 26 ознак": ("s_",),
    "лише вікно «до» (8)": ("s_ref_before_", "s_enr_before_"),
    "«під час», «після», відновлення (18)": ("s_ref_during_", "s_ref_after_", "s_ref_trec", "s_enr_during_",
                                              "s_enr_after_", "s_enr_trec"),
    "лише s_ref (13)": ("s_ref_",),
    "s_ref «до» (4)": ("s_ref_before_",),
    "s_ref «під час», «після», відновлення (9)": ("s_ref_during_", "s_ref_after_", "s_ref_trec"),
    "лише s_enr (13)": ("s_enr_",),
}


def ablation() -> dict:
    """LOSO логістичної регресії на підмножинах ознак: об'єднані AUC і EER та AUC кожного фолду."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import train
    from pad.metrics import auc, eer

    base = to_arrays(read_csv(ROOT / "data" / "features.csv"))
    out = {}
    for name, prefixes in ABLATION.items():
        names = [n for n in FEATURE_NAMES if n.startswith(prefixes)]
        data = dict(base)
        data["X"] = base["X"][:, [FEATURE_NAMES.index(n) for n in names]]
        res = train.loso(data, "logistic", {"C": 1.0, "seed": 42, "target_bpcer": 0.05, "names": names})
        rows = res["session_rows"]
        fold_auc = {}
        for subj in sorted({r["subject"] for r in rows}):
            rr = [r for r in rows if r["subject"] == subj]
            fold_auc[subj] = auc([r["score"] for r in rr], [r["label"] for r in rr])
        out[name] = {"n_features": len(names), "auc": res["session"]["auc"], "eer": res["session"]["eer"],
                     "fold_auc": fold_auc, "apcer": res["session"]["operating_point"]["apcer"],
                     "bpcer": res["session"]["operating_point"]["bpcer"],
                     "bootstrap_auc": bootstrap_auc(rows)}
    return out


def bootstrap_auc(rows: list[dict], n: int = 2000, seed: int = 0) -> list[float]:
    """95 % інтервал AUC бутстрепом за учасниками (сеанси одного учасника беруться разом)."""
    from pad.metrics import auc

    rng = np.random.default_rng(seed)
    subjects = sorted({r["subject"] for r in rows})
    by = {s: [r for r in rows if r["subject"] == s] for s in subjects}
    vals = []
    for _ in range(n):
        pick = [r for s in rng.choice(subjects, len(subjects)) for r in by[s]]
        y = [r["label"] for r in pick]
        if len(set(y)) == 2:
            vals.append(auc([r["score"] for r in pick], y))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def session_failures(recs: dict) -> dict:
    """Скільки сеансів мають хоча б один невиконаний виклик (у живому сеансі – повтор або «не пройдено»)."""
    out = {}
    for cls in ("bona_fide", "attack"):
        sess = [r for r in recs.values() if r["metadata"]["presentation_type"] == cls]
        failed = sum(any(not ch["performed"] for ch in r["decision"]["explanation"]["attempts"][0]) for r in sess)
        none = sum(all(not ch["performed"] for ch in r["decision"]["explanation"]["attempts"][0]) for r in sess)
        out[cls] = {"sessions": len(sess), "with_failed_challenge": failed, "no_performed": none}
    return out


def load_records(key: bytes) -> dict[str, dict]:
    recs = {}
    for f in sorted((ROOT / "data" / "recordings").glob("*/*/session.json.enc")):
        recs[f.parent.name] = read_json_encrypted(f, key)
    return recs


def challenge_rates(recs: dict) -> dict:
    done, total = Counter(), Counter()
    for rec in recs.values():
        cls = rec["metadata"]["presentation_type"]
        for ch in rec["decision"]["explanation"]["attempts"][0]:
            total[(cls, ch["challenge_type"])] += 1
            done[(cls, ch["challenge_type"])] += bool(ch["performed"])
    return {f"{c}:{t}": [done[(c, t)], total[(c, t)]] for c in ("bona_fide", "attack") for t in TYPES}


def fig_series(recs: dict, out: Path, driver: str = "P03", session: str = "s04") -> str:
    """s_ref(t) і s_enr(t) справжнього подання і атаки з тими самими рухами учасника."""
    pick = {}
    for name, rec in recs.items():
        meta = rec["metadata"]
        if meta["participant"] == driver and name.endswith("_" + session):
            pick[meta["presentation_type"]] = rec
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 5.6), sharex=True)
    for ax, (cls, title) in zip(axes, (("bona_fide", "а) справжнє подання"), ("attack", "б) атака підміни обличчя"))):
        expl = pick[cls]["decision"]["explanation"]
        t = np.array(expl["series"]["t"])
        for key, style, label in (("s_ref", "-", r"$s_{ref}(t)$"), ("s_enr", "--", r"$s_{enr}(t)$")):
            v = np.array([np.nan if x is None else x for x in expl["series"][key]], dtype=float)
            ax.plot(t, v, style, color="black", lw=1.1, label=label)
        for ch in expl["attempts"][0]:
            a, b = ch["t_start"] - expl["t0"], ch["t_end"] - expl["t0"]
            ax.axvspan(a, b, color="0.85", lw=0)
            ax.text((a + b) / 2, 1.02, TYPE_UA[ch["challenge_type"]], ha="center", va="bottom", fontsize=10)
        ax.set_ylim(-0.2, 1.15)
        ax.set_ylabel("Подібність")
        ax.set_title(title, loc="left", pad=16)
        ax.yaxis.set_major_formatter(COMMA)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=10)
    axes[1].set_xlabel("Час від початку сеансу, с")
    axes[1].xaxis.set_major_formatter(COMMA)
    fig.tight_layout()
    path = out / "fig_series.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return f"{driver} {session}"


def fig_det(out: Path) -> None:
    from scipy.stats import norm
    from sklearn.metrics import det_curve

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    eps = 2e-3
    for (run, model), (label, style) in {
        ("pilot_all", "logistic"): ("логістична регресія, 26 ознак (EER 0)", "-"),
        ("pilot_sref", "logistic"): (r"логістична регресія, лише $s_{ref}$", "--"),
        ("pilot_all", "threshold"): (r"поріг на мінімумі $s_{ref}$", ":"),
    }.items():
        rows = sessions(run, model)
        y = [int(r["label"]) for r in rows]
        s = [float(r["score"]) for r in rows]
        bp, ap, _ = det_curve(y, s)
        # без помилок крива вироджується в точку (0; 0) – її позначено маркером у лівому нижньому куті
        ax.plot(norm.ppf(np.clip(bp, eps, 1 - eps)), norm.ppf(np.clip(ap, eps, 1 - eps)), style,
                color="black", lw=1.3, label=label, marker="o" if len(bp) <= 2 else None, ms=6)
    ticks = np.array([0.005, 0.02, 0.05, 0.2, 0.5, 0.8])
    ax.set_xticks(norm.ppf(ticks), [f"{v:g}".replace(".", ",") for v in ticks])
    ax.set_yticks(norm.ppf(ticks), [f"{v:g}".replace(".", ",") for v in ticks])
    # крива без помилок (EER 0) проходить уздовж нижньої і лівої меж – межі осей трохи ширші за eps
    ax.set_xlim(norm.ppf(eps / 2), norm.ppf(0.95))
    ax.set_ylim(norm.ppf(eps / 2), norm.ppf(0.9995))
    ax.set_xlabel("BPCER")
    ax.set_ylabel("APCER")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig_det.png", dpi=200)
    plt.close(fig)


def fig_challenges(rates: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    x = np.arange(len(TYPES))
    for off, cls, hatch, label in ((-0.2, "bona_fide", "", "справжні подання"), (0.2, "attack", "//", "атаки")):
        vals = [rates[f"{cls}:{t}"][0] / max(rates[f"{cls}:{t}"][1], 1) for t in TYPES]
        bars = ax.bar(x + off, vals, 0.38, color="white", edgecolor="black", hatch=hatch, label=label)
        for b, t in zip(bars, TYPES):
            d, n = rates[f"{cls}:{t}"]
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{d}/{n}", ha="center", fontsize=9)
    ax.set_xticks(x, [TYPE_UA2[t] for t in TYPES])
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Частка виконаних")
    ax.yaxis.set_major_formatter(COMMA)
    ax.legend(fontsize=10, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_challenges.png", dpi=200)
    plt.close(fig)


def main() -> int:
    setup_console()
    out = ROOT / "runs" / "pilot_report"
    out.mkdir(parents=True, exist_ok=True)
    recs = load_records(load_key())
    rates = challenge_rates(recs)
    report = {
        "summary": summary_table(),
        "per_subject": per_subject(),
        "univariate_auc": univariate_auc(),
        "challenge_rates": rates,
        "n_sessions": dict(Counter(r["metadata"]["presentation_type"] for r in recs.values())),
        "series_example": fig_series(recs, out),
        "session_failures": session_failures(recs),
        "ablation": ablation(),
    }
    fig_det(out)
    fig_challenges(rates, out)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float),
                                      encoding="utf-8")
    for k, v in report["summary"].items():
        print(f"{k:38s} APCER {v['apcer']:.3f} [{v['apcer_ci'][0]:.2f}; {v['apcer_ci'][1]:.2f}]  "
              f"BPCER {v['bpcer']:.3f} [{v['bpcer_ci'][0]:.2f}; {v['bpcer_ci'][1]:.2f}]  "
              f"BPCER20 {v['bpcer20']:.3f}  EER {v['eer']:.3f}  AUC {v['auc']:.3f}")
    for k, v in report["ablation"].items():
        print(f"{k:45s} AUC {v['auc']:.3f} [{v['bootstrap_auc'][0]:.2f}; {v['bootstrap_auc'][1]:.2f}] "
              f"EER {v['eer']:.3f}  фолди {', '.join(f'{a:.2f}' for a in v['fold_auc'].values())}")
    print("сеанси з невиконаним викликом:", report["session_failures"])
    print("викликів виконано:", {k: f"{d}/{n}" for k, (d, n) in rates.items()})
    print("результати:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
