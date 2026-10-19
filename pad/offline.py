"""Офлайн-обробка записаного відео за анотацією з часовими мітками викликів.

Використовується для записів, зроблених без прототипу: віддалений запис
учасника у браузері (remote/remote_recorder.html) і атаки, згенеровані з цих
записів офлайн (scripts/generate_attacks.py). Кожен кадр проходить той самий
конвеєр, що й у живому сеансі (детекція, Face Mesh, Hands, ембединг), а
виклики перевіряються тими самими класами Challenge. Результат – запис у
форматі session.json живого сеансу, тому scripts/build_dataset.py обробляє
обидва джерела однаково.

Формат анотації (frs-pad-remote/1), часи в секундах від початку відео:
    {"format": "frs-pad-remote/1", "participant": "P02", "consent": true,
     "conditions": {"lighting": "...", "webcam": "...", "source": "remote_browser"},
     "enroll": {"t_start": 1.0, "t_end": 4.0},
     "sessions": [{"reference": [t0, t1],
                   "challenges": [{"challenge_type": "cover_left", "t_start": .., "t_end": ..},
                                  {"challenge_type": "turn_head", "direction": "left", ...}]}]}
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .challenges import Challenge, HandSwipeChallenge, HeadTurnChallenge, OcclusionChallenge
from .features import DriftFeatureExtractor, reference_embedding
from .pipeline import FramePipeline, enroll_embedding
from .settings import CHALLENGE_TYPES, PadSettings
from .structures import FrameData

ANNOTATION_FORMAT = "frs-pad-remote/1"
HAND_TYPES = {"cover_left", "cover_right", "cover_mouth", "hand_swipe"}


class AnnotationError(ValueError):
    """Анотація не відповідає формату frs-pad-remote/1."""


def validate_annotation(ann: dict) -> None:
    """Перевіряє обов'язкові поля, типи викликів і порядок часових міток."""
    if ann.get("format") != ANNOTATION_FORMAT:
        raise AnnotationError(f"очікується format = {ANNOTATION_FORMAT}")
    if not ann.get("participant"):
        raise AnnotationError("немає псевдоніма учасника (participant)")
    if ann.get("consent") is not True:
        raise AnnotationError("учасник не підтвердив згоду (consent)")
    sessions = ann.get("sessions") or []
    if not sessions:
        raise AnnotationError("немає жодного сеансу (sessions)")
    for i, s in enumerate(sessions, 1):
        ref = s.get("reference")
        if not ref or len(ref) != 2 or ref[1] <= ref[0]:
            raise AnnotationError(f"сеанс {i}: некоректне вікно еталона")
        prev = ref[1]
        for ch in s.get("challenges") or []:
            if ch.get("challenge_type") not in CHALLENGE_TYPES:
                raise AnnotationError(f"сеанс {i}: невідомий тип виклику {ch.get('challenge_type')}")
            if ch["challenge_type"] == "turn_head" and ch.get("direction") not in ("left", "right"):
                raise AnnotationError(f"сеанс {i}: для turn_head потрібен direction left або right")
            if not (prev - 1e-6 <= ch["t_start"] < ch["t_end"]):
                raise AnnotationError(f"сеанс {i}: часові мітки викликів не впорядковані")
            prev = ch["t_end"]


def challenge_from_annotation(ch: dict, settings: PadSettings) -> Challenge:
    """Виклик з параметрами settings і типом (напрямком) з анотації."""
    common = {"duration_s": ch["t_end"] - ch["t_start"],
              "consecutive_frames": settings.consecutive_frames,
              "face_hold_s": settings.face_hold_s}
    kind = ch["challenge_type"]
    if kind.startswith("cover_"):
        return OcclusionChallenge(kind.removeprefix("cover_"), settings.min_coverage, **common)
    if kind == "turn_head":
        return HeadTurnChallenge(ch["direction"], settings.min_yaw_deg, **common)
    return HandSwipeChallenge(settings.min_shift, **common)


@dataclass
class VideoFrames:
    """Дані всіх кадрів відео після конвеєра."""

    frames: list[FrameData] = field(default_factory=list)
    fps: float = 0.0
    size: tuple[int, int] = (0, 0)

    def between(self, t0: float, t1: float) -> list[FrameData]:
        return [f for f in self.frames if t0 <= f.t <= t1]


def _needs_hands(t: float, ann: dict) -> bool:
    for s in ann["sessions"]:
        for ch in s.get("challenges") or []:
            if ch["challenge_type"] in HAND_TYPES and ch["t_start"] - 0.5 <= t <= ch["t_end"] + 0.5:
                return True
    return False


def run_pipeline(video: str | Path, pipeline: FramePipeline, ann: dict | None = None,
                 progress=None) -> VideoFrames:
    """Проганяє всі кадри відео через конвеєр; час кадру – мітка часу кадру у файлі."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"не вдалося відкрити відео {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    out = VideoFrames(fps=fps)
    i = 0
    prev_t = -1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # записи MediaRecorder (webm) мають змінну частоту кадрів: беремо мітку часу кадру
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        t = pos if pos > prev_t and (i > 0 or pos >= 0) and pos < 1e5 else i / fps
        if t <= prev_t:
            t = prev_t + 1.0 / fps
        prev_t = t
        hands = True if ann is None else _needs_hands(t, ann)
        fd, _ = pipeline.process(frame, t, landmarks=True, hands=hands)
        out.frames.append(fd)
        out.size = (frame.shape[1], frame.shape[0])
        i += 1
        if progress and i % 30 == 0:
            progress(i, total)
    cap.release()
    return out


def enrolled_from_video(vf: VideoFrames, ann: dict, min_score: float = 0.9) -> np.ndarray | None:
    """Ембединг учасника з вікна реєстрації (enroll) анотації."""
    enroll = ann.get("enroll")
    if not enroll:
        return None
    return enroll_embedding(vf.between(enroll["t_start"], enroll["t_end"]), min_score)


def face_ratio(vf: VideoFrames, ann: dict) -> float:
    """Частка кадрів поза викликами з перекриттям, на яких знайдено обличчя (контроль якості)."""
    covered = [(ch["t_start"], ch["t_end"]) for s in ann["sessions"] for ch in s.get("challenges") or []
               if ch["challenge_type"] in HAND_TYPES]
    frames = [f for f in vf.frames if not any(a <= f.t <= b for a, b in covered)]
    if not frames:
        return 0.0
    return sum(1 for f in frames if f.embedding is not None and f.det_score > 0) / len(frames)


def session_records(vf: VideoFrames, ann: dict, settings: PadSettings, enrolled: np.ndarray | None,
                    metadata: dict) -> list[dict]:
    """Записи у форматі session.json живого сеансу – по одному на сеанс анотації."""
    from .verifier import ChallengeVerifier

    features = DriftFeatureExtractor(settings.before_s, settings.after_s, settings.recovery_ratio)
    verifier = ChallengeVerifier(mirrored=False)
    records = []
    for idx, sess in enumerate(ann["sessions"], 1):
        r0, r1 = sess["reference"]
        chs = sess.get("challenges") or []
        t_last = (chs[-1]["t_end"] if chs else r1) + settings.after_s
        ref_frames = vf.between(r0, r1)
        reference = reference_embedding([f.embedding for f in ref_frames], [f.det_score for f in ref_frames],
                                        settings.min_det_score)
        series_frames = vf.between(r0, t_last)
        t = np.array([f.t for f in series_frames], dtype=float)
        enr = enrolled if enrolled is not None else reference
        if reference is not None:
            s_ref = [features.similarity(f.embedding, reference) for f in series_frames]
            s_enr = [features.similarity(f.embedding, enr) for f in series_frames]
        else:
            s_ref = s_enr = [float("nan")] * len(series_frames)
        summaries = []
        for ch in chs:
            c = challenge_from_annotation(ch, settings)
            for f in vf.between(ch["t_start"], ch["t_end"]):
                c.add_frame(f, verifier)
            c.t_start, c.t_end = ch["t_start"], ch["t_end"]
            summaries.append(c.summary())
        t0 = r0
        records.append({
            "metadata": {**copy.deepcopy(metadata), "session_index": idx},
            "settings": settings.to_dict(),
            "decision": {"decision": "offline", "score": None, "explanation": {
                "claimed_person_id": metadata.get("claimed_identity"),
                "attempts": [summaries],
                "reference_found": reference is not None,
                "t0": t0,
                "series": {"t": [round(float(v), 4) for v in t - t0],
                           "s_ref": [round(float(v), 4) if np.isfinite(v) else None for v in s_ref],
                           "s_enr": [round(float(v), 4) if np.isfinite(v) else None for v in s_enr]},
            }},
        })
    verifier.close()
    return records
