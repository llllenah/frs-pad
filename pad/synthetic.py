"""Синтетичні дані для тестів і перевірки скрипта навчання.

Містить генератори ключових точок обличчя та кисті (з відомими кутами
повороту і перекриттям), ембедингів із заданою косинусною подібністю та
рядів s_ref(t), s_enr(t) для справжніх подань і атак. Синтетичні ряди лише
відтворюють якісну різницю, описану в підрозділі 3.3 (плавний провал і швидке
відновлення для справжнього обличчя, різкі стрибки і повільне відновлення для
підміненого), і не замінюють пілотний набір даних.
"""

from __future__ import annotations

import math
import random

import numpy as np

from .challenges import ChallengeGenerator
from .features import FEATURE_NAMES, DriftFeatureExtractor, l2_normalize
from .settings import PadSettings
from .verifier import CHEEK_IMG_LEFT, CHEEK_IMG_RIGHT, CHIN, FOREHEAD, NOSE_TIP, POSE_IDX, POSE_MODEL

# ----- ключові точки -----


def rotation(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Матриця повороту голови з тими самими знаками кутів, що й head_pose()."""
    y, p, r = map(math.radians, (yaw, pitch, roll))
    ry = np.array([[math.cos(-y), 0, math.sin(-y)], [0, 1, 0], [-math.sin(-y), 0, math.cos(-y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(-p), -math.sin(-p)], [0, math.sin(-p), math.cos(-p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _face_cloud(seed: int = 0) -> np.ndarray:
    """468 точок узагальненого обличчя в системі моделі POSE_MODEL."""
    rng = np.random.default_rng(seed)
    n = 468
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = np.sqrt(rng.uniform(0, 1, n))
    rad[: n // 3] = 1.0  # третина точок на контурі овалу
    pts = np.stack([330 * rad * np.cos(ang), -20 + 400 * rad * np.sin(ang),
                    150 + 60 * rad ** 2], axis=1)
    pts[FOREHEAD] = [0, -420, 210]
    pts[CHIN] = [0, 380, 120]
    pts[CHEEK_IMG_LEFT] = [-330, 0, 210]
    pts[CHEEK_IMG_RIGHT] = [330, 0, 210]
    for i, idx in enumerate(POSE_IDX):
        pts[idx] = POSE_MODEL[i]
    pts[NOSE_TIP] = POSE_MODEL[0]
    return pts


_CLOUD = _face_cloud()


def synthetic_face(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
                   size: tuple[int, int] = (1280, 720), center: tuple[float, float] | None = None,
                   face_width_px: float = 260.0, noise_px: float = 0.0,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Точки обличчя (468, 2), отримані проєкцією 3D-моделі з камерою f = ширина кадру."""
    w, h = size
    cx, cy = center if center is not None else (w / 2, h / 2)
    f = float(w)
    z0 = f * 660.0 / face_width_px
    cam = _CLOUD @ rotation(yaw, pitch, roll).T
    # зсув так, щоб центр обличчя (а не кінчик носа) проєктувався в (cx, cy)
    tx = (cx - w / 2) * z0 / f
    ty = (cy - h / 2) * z0 / f
    cam = cam + np.array([tx, ty, z0])
    pts = np.stack([f * cam[:, 0] / cam[:, 2] + w / 2, f * cam[:, 1] / cam[:, 2] + h / 2], axis=1)
    if noise_px > 0:
        pts = pts + (rng or np.random.default_rng()).normal(0, noise_px, pts.shape)
    return pts.astype(np.float32)


def synthetic_hand(center: tuple[float, float], width: float, height: float) -> np.ndarray:
    """21 точка кисті з розкритою долонею: зап'ясток унизу, кінчики пальців угорі."""
    cx, cy = center
    pts = [(0.0, 0.5)]  # зап'ясток
    for k in range(1, 5):  # великий палець
        pts.append((-0.25 - 0.06 * k, 0.4 - 0.12 * k))
    for col, x in enumerate((-0.3, -0.1, 0.1, 0.3)):  # вказівний – мізинець
        for k in range(4):
            pts.append((x, 0.1 - 0.2 * k))
    arr = np.array(pts) * np.array([width, height]) + np.array([cx, cy])
    return arr.astype(np.float32)


def hand_over_face(face: np.ndarray, side: str, scale: float = 1.0) -> np.ndarray:
    """Кисть, що закриває ліву (з погляду користувача) чи праву половину обличчя або рот.

    Кадр не віддзеркалений: ліва половина обличчя користувача – праворуч у кадрі.
    """
    fw = float(np.linalg.norm(face[CHEEK_IMG_RIGHT] - face[CHEEK_IMG_LEFT]))
    fh = float(np.linalg.norm(face[CHIN] - face[FOREHEAD]))
    nose = face[NOSE_TIP]
    if side == "left":
        return synthetic_hand((nose[0] + 0.25 * fw, nose[1]), 0.6 * fw * scale, 1.0 * fh * scale)
    if side == "right":
        return synthetic_hand((nose[0] - 0.25 * fw, nose[1]), 0.6 * fw * scale, 1.0 * fh * scale)
    if side == "mouth":
        return synthetic_hand((nose[0], nose[1] + 0.3 * fh), 1.2 * fw * scale, 0.45 * fh * scale)
    raise ValueError(side)


# ----- ембединги -----


def embedding_with_similarity(ref: np.ndarray, s: float, rng: np.random.Generator) -> np.ndarray:
    """Одиничний вектор e такий, що e · ref = s."""
    ref = l2_normalize(ref)
    u = rng.normal(size=ref.shape).astype(np.float32)
    u -= np.dot(u, ref) * ref
    u = l2_normalize(u)
    s = float(np.clip(s, -1.0, 1.0))
    return l2_normalize(s * ref + math.sqrt(max(0.0, 1 - s * s)) * u)


# ----- ряди подібності -----

DIP_DEPTH = {"cover_left": (0.25, 0.60), "cover_right": (0.25, 0.60),
             "cover_mouth": (0.20, 0.50), "turn_head": (0.15, 0.45), "hand_swipe": (0.10, 0.35)}


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def simulate_challenge_series(kind: str, attack: bool, rng: np.random.Generator,
                              subject: dict | None = None, fps: float = 30.0,
                              before_s: float = 1.0, after_s: float = 1.5) -> dict:
    """Ряди t, s_ref, s_enr для одного виклику (t = 0 – початок, t_end – кінець виклику)."""
    subject = subject or {"k_enr": 0.75, "noise": 0.012}
    during = rng.uniform(1.2, 3.0) + 0.5 if kind != "hand_swipe" else rng.uniform(0.8, 2.0)
    t = np.arange(-before_s, during + after_s + 1e-9, 1.0 / fps)
    t = t[rng.uniform(size=len(t)) > 0.03]  # пропущені кадри
    noise = subject["noise"] * (1.3 if attack else 1.0)
    base = rng.uniform(0.93, 0.97) if not attack else rng.uniform(0.91, 0.96)
    depth = rng.uniform(*DIP_DEPTH[kind])

    # плавний провал: рука заходить, тримається, виходить уже після кінця виклику
    t_in = rng.uniform(0.3, 0.8)
    if kind == "hand_swipe":
        center, width = during * rng.uniform(0.4, 0.7), rng.uniform(0.25, 0.5)
        dip = depth * np.exp(-0.5 * ((t - center) / (width / 2)) ** 2)
        release = during
    else:
        release = during + rng.uniform(0.1, 0.5)
        dip = depth * _smoothstep(t / t_in) * (1 - _smoothstep((t - release) / 0.3))
    s_ref = base - dip

    # для атаки відновлення повільніше, іноді неповне (генератор не повернувся до цілі)
    hard = attack and rng.uniform() < 0.3  # «вдала» атака без збоїв генератора
    if attack and not hard:
        tau = rng.uniform(0.3, 1.2)
        level = rng.uniform(0.80, 0.88) if rng.uniform() < 0.35 else 1.0
        rec = base * level - (base * level - (base - depth)) * np.exp(-(t - release) / tau)
        s_ref = np.where(t > release, rec, s_ref)

    s_enr_scale = subject["k_enr"] / 0.96 * (rng.uniform(0.75, 1.0) if attack else 1.0)
    s_enr = s_ref * s_enr_scale

    # збої генератора під час перекриття: різкі стрибки вниз на кілька кадрів
    if attack and not hard:
        n_glitch = rng.integers(1, 5)
        during_idx = np.nonzero((t >= 0) & (t <= release))[0]
        for _ in range(n_glitch if len(during_idx) else 0):
            start = rng.choice(during_idx)
            length = rng.integers(2, 9)
            sel = slice(start, min(start + length, len(t)))
            s_ref[sel] -= rng.uniform(0.15, 0.4)
            s_enr[sel] = rng.uniform(0.1, 0.25, size=len(t[sel]))

    # у справжніх поданнях теж бувають короткі провали: розмиття рукою, що швидко рухається
    if not attack and rng.uniform() < 0.3:
        start = rng.integers(0, len(t))
        sel = slice(start, min(start + rng.integers(1, 3), len(t)))
        drop = rng.uniform(0.1, 0.3)
        s_ref[sel] -= drop
        s_enr[sel] -= drop * s_enr_scale

    s_ref = np.clip(s_ref + rng.normal(0, noise, len(t)), -1, 1)
    s_enr = np.clip(s_enr + rng.normal(0, noise, len(t)), -1, 1)
    return {"t": t, "s_ref": s_ref, "s_enr": s_enr, "t_start": 0.0, "t_end": during}


def make_synthetic_dataset(n_subjects: int = 6, sessions_per_class: int = 8, seed: int = 42,
                           settings: PadSettings | None = None,
                           species: str = "synthetic_swap") -> list[dict]:
    """Синтетичний набір: рядки з ознаками викликів, мітками і учасниками.

    Кожен учасник має сеанси справжніх подань і атак; у сеансі – challenge_count
    викликів, згенерованих тим самим ChallengeGenerator, що й у прототипі.
    """
    settings = settings or PadSettings()
    rng = np.random.default_rng(seed)
    gen = ChallengeGenerator(settings, rng=random.Random(seed))
    extractor = DriftFeatureExtractor(settings.before_s, settings.after_s, settings.recovery_ratio)
    rows: list[dict] = []
    for si in range(n_subjects):
        subject = {"id": f"S{si + 1:02d}", "k_enr": rng.uniform(0.65, 0.85),
                   "noise": rng.uniform(0.008, 0.02)}
        for label in (0, 1):
            for k in range(sessions_per_class):
                session_uid = f"{subject['id']}_{'atk' if label else 'bf'}_{k:02d}"
                for order, kind in enumerate(gen.generate_types(settings.challenge_count)):
                    ser = simulate_challenge_series(kind, bool(label), rng, subject,
                                                    before_s=settings.before_s,
                                                    after_s=settings.after_s)
                    fv = extractor.extract_series(ser["t"], ser["s_ref"], ser["s_enr"],
                                                  ser["t_start"], ser["t_end"])
                    rows.append({"subject": subject["id"], "session_uid": session_uid,
                                 "label": label, "pai_species": species if label else "",
                                 "challenge_type": kind, "order": order,
                                 **dict(zip(FEATURE_NAMES, map(float, fv.values)))})
    return rows
