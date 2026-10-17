"""Показники якості виявлення атак за ДСТУ ISO/IEC 30107-3:2023.

Оцінка score – чим більша, тим імовірніша атака; подання класифікується як
атака, якщо score >= τ. Мітка label = 1 – атака, 0 – справжнє подання.

APCER – частка атак, класифікованих як справжні подання (окремо для кожного
виду атаки, PAI species, і найбільше значення серед них); BPCER – частка
справжніх подань, класифікованих як атаки; BPCER20 – BPCER за APCER = 5 %.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _split(scores, labels):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(int)
    return s, y, s[y == 0], s[y == 1]


def apcer_bpcer(scores, labels, threshold: float, species=None) -> dict:
    """APCER (для кожного виду атаки і найбільший), BPCER і ACER за порогом τ."""
    s, y, bona, attack = _split(scores, labels)
    bpcer = float(np.mean(bona >= threshold)) if len(bona) else float("nan")
    per_species: dict[str, float] = {}
    if len(attack):
        sp = np.asarray(species if species is not None else ["attack"] * len(s), dtype=object)
        for name in sorted(set(sp[y == 1])):
            sel = (y == 1) & (sp == name)
            per_species[str(name)] = float(np.mean(s[sel] < threshold))
    apcer = max(per_species.values()) if per_species else float("nan")
    return {"threshold": float(threshold), "apcer": apcer, "apcer_per_species": per_species,
            "bpcer": bpcer, "acer": (apcer + bpcer) / 2}


def rates_from_decisions(is_attack_pred, labels, species=None) -> dict:
    """APCER, BPCER і ACER за готовими рішеннями (коли пороги різні для різних фолдів)."""
    pred = np.asarray(is_attack_pred, dtype=bool)
    y = np.asarray(labels).astype(int)
    bpcer = float(np.mean(pred[y == 0])) if (y == 0).any() else float("nan")
    sp = np.asarray(species if species is not None else ["attack"] * len(y), dtype=object)
    per_species = {str(n): float(np.mean(~pred[(y == 1) & (sp == n)])) for n in sorted(set(sp[y == 1]))}
    apcer = max(per_species.values()) if per_species else float("nan")
    return {"apcer": apcer, "apcer_per_species": per_species, "bpcer": bpcer,
            "acer": (apcer + bpcer) / 2}


def _thresholds(s: np.ndarray) -> np.ndarray:
    return np.concatenate([np.unique(s), [np.inf]])


def threshold_for_bpcer(scores, labels, target: float = 0.05) -> float:
    """Найменший поріг τ, за якого BPCER не перевищує target (підбір на валідації)."""
    _, _, bona, _ = _split(scores, labels)
    if len(bona) == 0:
        return 0.5
    for thr in _thresholds(bona):
        if np.mean(bona >= thr) <= target:
            return float(thr) if np.isfinite(thr) else float(bona.max() + 1e-6)
    return float(bona.max() + 1e-6)


def bpcer_at_apcer(scores, labels, apcer_target: float = 0.05) -> tuple[float, float]:
    """BPCER за найбільшого порогу, де APCER не перевищує apcer_target.

    Для apcer_target = 0,05 це BPCER20. Повертає (BPCER, τ).
    """
    s, _, bona, attack = _split(scores, labels)
    if len(bona) == 0 or len(attack) == 0:
        return float("nan"), float("nan")
    best_thr = -np.inf
    for thr in _thresholds(s):
        if np.mean(attack < thr) <= apcer_target:
            best_thr = thr
    if not np.isfinite(best_thr):
        best_thr = float(attack.min())
    return float(np.mean(bona >= best_thr)), float(best_thr)


def eer(scores, labels) -> tuple[float, float]:
    """Рівень рівних помилок EER (APCER = BPCER) і відповідний поріг."""
    s, y, bona, attack = _split(scores, labels)
    if len(bona) == 0 or len(attack) == 0:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, s)  # fpr = BPCER, 1 – tpr = APCER
    fnr = 1 - tpr
    diff = fpr - fnr
    i = int(np.nonzero(diff >= 0)[0][0])
    if i == 0 or diff[i] == 0:
        return float((fpr[i] + fnr[i]) / 2), float(thr[i])
    # лінійна інтерполяція між точками i – 1 та i
    a = diff[i - 1] / (diff[i - 1] - diff[i])
    e = fpr[i - 1] + a * (fpr[i] - fpr[i - 1])
    t = thr[i - 1] + a * (thr[i] - thr[i - 1]) if np.isfinite(thr[i - 1]) else thr[i]
    return float(e), float(t)


def auc(scores, labels) -> float:
    """Площа під ROC-кривою."""
    _, y, bona, attack = _split(scores, labels)
    if len(bona) == 0 or len(attack) == 0:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def evaluate(scores, labels, threshold: float, species=None) -> dict:
    """Усі показники для набору оцінок: APCER, BPCER, ACER, BPCER20, EER, AUC."""
    res = apcer_bpcer(scores, labels, threshold, species)
    res["bpcer20"], res["bpcer20_threshold"] = bpcer_at_apcer(scores, labels, 0.05)
    res["eer"], res["eer_threshold"] = eer(scores, labels)
    res["auc"] = auc(scores, labels)
    res["n_bona_fide"] = int(np.sum(np.asarray(labels) == 0))
    res["n_attack"] = int(np.sum(np.asarray(labels) == 1))
    return res


def error_curves(scores, labels) -> dict:
    """Точки ROC- і DET-кривих: BPCER і APCER для всіх порогів."""
    _, y, _, _ = _split(scores, labels)
    fpr, tpr, thr = roc_curve(y, np.asarray(scores, dtype=float))
    return {"bpcer": fpr, "apcer": 1 - tpr, "tpr": tpr, "threshold": thr}
