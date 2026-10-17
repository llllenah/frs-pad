"""Ознаки дрейфу ембединга ідентичності (DriftFeatureExtractor).

Для кожного кадру обчислюються s_ref(t) і s_enr(t) – косинусна подібність
ембединга кадру до еталона та до ембединга заявленої особи, формула (3.2).
Ряди розбиваються на вікна «до», «під час» і «після» виклику; у кожному
вікні обчислюються мінімум, середнє, дисперсія і найбільший міжкадровий
стрибок, а також час відновлення за формулою (3.4). Разом
2 × (3 × 4 + 1) = 26 ознак на виклик.
"""

from __future__ import annotations

import numpy as np

from .structures import FeatureVector

SERIES = ("ref", "enr")
WINDOWS = ("before", "during", "after")
STATS = ("min", "mean", "var", "jump")

#: Назви 26 ознак у фіксованому порядку.
FEATURE_NAMES: list[str] = [
    name
    for s in SERIES
    for name in [f"s_{s}_{w}_{st}" for w in WINDOWS for st in STATS] + [f"s_{s}_trec"]
]


class InsufficientDataError(ValueError):
    """У вікні замало кадрів з ембедингом, щоб обчислити ознаки."""


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Нормує вектор до одиничної довжини."""
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def reference_embedding(embeddings: list[np.ndarray], scores: list[float],
                        min_score: float = 0.9) -> np.ndarray | None:
    """Еталонний ембединг – нормований середній вектор, формула (3.1).

    Використовуються лише кадри з упевненістю детекції не менше min_score.
    """
    good = [l2_normalize(e) for e, s in zip(embeddings, scores) if e is not None and s >= min_score]
    if not good:
        return None
    return l2_normalize(np.sum(good, axis=0))


class DriftFeatureExtractor:
    """Обчислює 26 ознак дрейфу ембединга для одного виклику."""

    def __init__(self, before_s: float = 1.0, after_s: float = 1.5,
                 recovery_ratio: float = 0.9, min_samples: int = 2):
        self.before_s = before_s
        self.after_s = after_s
        self.recovery_ratio = recovery_ratio
        self.min_samples = min_samples

    @staticmethod
    def similarity(e: np.ndarray | None, ref: np.ndarray | None) -> float:
        """Косинусна подібність нормованих векторів (скалярний добуток)."""
        if e is None or ref is None:
            return float("nan")
        return float(np.dot(l2_normalize(e), l2_normalize(ref)))

    @staticmethod
    def window_stats(series: np.ndarray) -> dict[str, float]:
        """Мінімум, середнє, дисперсія і найбільший міжкадровий стрибок |s(t) – s(t – 1)|."""
        s = np.asarray(series, dtype=float)
        s = s[np.isfinite(s)]
        if len(s) == 0:
            raise InsufficientDataError("порожнє вікно")
        jump = float(np.max(np.abs(np.diff(s)))) if len(s) > 1 else 0.0
        return {"min": float(s.min()), "mean": float(s.mean()), "var": float(s.var()), "jump": jump}

    def recovery_time(self, t: np.ndarray, s: np.ndarray, t_end: float, s_before: float) -> float:
        """Час відновлення t_rec = min{t >= t_end : s(t) >= 0,9 · s_до} – t_end, формула (3.4).

        Якщо у вікні «після» подібність не відновилася, повертається тривалість вікна.
        """
        mask = (t >= t_end) & (t <= t_end + self.after_s) & np.isfinite(s)
        hit = np.nonzero(s[mask] >= self.recovery_ratio * s_before)[0]
        if len(hit) == 0:
            return float(self.after_s)
        return float(t[mask][hit[0]] - t_end)

    def windows(self, t: np.ndarray, t_start: float, t_end: float) -> dict[str, np.ndarray]:
        """Маски вікон «до», «під час» і «після» для ряду з часовими мітками t."""
        return {
            "before": (t >= t_start - self.before_s) & (t < t_start),
            "during": (t >= t_start) & (t <= t_end),
            "after": (t > t_end) & (t <= t_end + self.after_s),
        }

    def extract_series(self, t, s_ref, s_enr, t_start: float, t_end: float) -> FeatureVector:
        """Обчислює ознаки за рядами подібності та межами виклику."""
        t = np.asarray(t, dtype=float)
        rows = {"ref": np.asarray(s_ref, dtype=float), "enr": np.asarray(s_enr, dtype=float)}
        masks = self.windows(t, t_start, t_end)
        values: list[float] = []
        for name in SERIES:
            s = rows[name]
            stats = {}
            for w in WINDOWS:
                part = s[masks[w]]
                if np.isfinite(part).sum() < self.min_samples:
                    raise InsufficientDataError(f"у вікні «{w}» замало кадрів для s_{name}")
                stats[w] = self.window_stats(part)
                values.extend(stats[w][st] for st in STATS)
            values.append(self.recovery_time(t, s, t_end, stats["before"]["mean"]))
        return FeatureVector(np.array(values, dtype=np.float32), list(FEATURE_NAMES))

    def extract(self, session, challenge) -> FeatureVector:
        """Ознаки виклику challenge за рядами подібності сеансу session."""
        t, s_ref, s_enr = session.series_arrays()
        return self.extract_series(t, s_ref, s_enr, challenge.t_start, challenge.t_end)
