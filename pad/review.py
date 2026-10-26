"""Перегляд сеансу: відео, ряди s_ref(t), s_enr(t), вікна викликів і рішення.

Записи набору даних зашифровано (pad/storage.py), а браузер не відтворює mp4v і
AVI (MJPG), тому сервер розшифровує відео лише в пам'яті, декодує OpenCV тільки
кадри потрібного сеансу, зменшує їх до REVIEW_WIDTH пікселів і стискає в JPEG.
Кадри тримаються в пам'яті сервера (невеликий кеш LRU) і віддаються браузеру
по одному. Відкритих файлів на диску не створюється.

Два формати записів:
    живий запис (веб-інтерфейс, record_session.py):
        <запис>/video.mp4.enc, session.json.enc; кадр i відповідає series.t[i];
    імпортований запис (scripts/import_remote.py):
        <запис>/video.avi.enc, frame_times.json.enc (абсолютний час кожного кадру),
        annotation.json.enc і теки сеансів <запис>_sNN/session.json.enc, де
        explanation.t0 – початок сеансу в тому самому часі, а series.t – від t0.
"""

from __future__ import annotations

import io
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .classifiers import session_score
from .features import DriftFeatureExtractor, InsufficientDataError
from .storage import SUFFIX, DecryptionError, read_encrypted, read_encrypted_buffer, read_json_encrypted

log = logging.getLogger(__name__)

#: Ширина кадрів перегляду, пікселів, і якість JPEG.
REVIEW_WIDTH = 640
JPEG_QUALITY = 75
#: Найбільше кадрів одного сеансу; довші сеанси проріджуються рівномірно.
MAX_FRAMES = 1800

#: Ідентифікатор сеансу в URL: ім'я теки запису або код сеансу VS-…
UID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

#: Короткі назви викликів для графіка і списку.
CHALLENGE_NAMES = {
    "cover_left": "Закрити ліву половину обличчя",
    "cover_right": "Закрити праву половину обличчя",
    "cover_mouth": "Закрити рот і підборіддя",
    "turn_head": "Поворот голови",
    "hand_swipe": "Провести рукою перед обличчям",
}
CHALLENGE_SHORT = {
    "cover_left": "ліва половина", "cover_right": "права половина", "cover_mouth": "рот",
    "hand_swipe": "рука перед обличчям",
}

FAILURE_TEXT = {
    "timeout": "перевищено тривалість сеансу",
    "no_face": "обличчя не виявлено під час фіксації еталона",
    "not_performed": "виклики не виконано",
    "error": "збій модуля",
}


class ReviewError(Exception):
    """Сеанс неможливо показати; status – код HTTP для відповіді."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


@dataclass
class SessionRef:
    """Сеанс у data/recordings."""

    uid: str
    folder: Path            # тека з session.json.enc
    recording: Path         # тека запису з відео
    layout: str             # "live" або "imported"

    @property
    def annotation(self) -> Path:
        return self.folder / ("session.json" + SUFFIX)


@dataclass
class ReviewData:
    """Підготовлений сеанс: метадані для інтерфейсу і кадри JPEG у пам'яті."""

    meta: dict
    frames: list[bytes] = field(default_factory=list)


# ----- пошук сеансів -----

def find_sessions(dirs: list[Path]) -> list[SessionRef]:
    """Сеанси в теках записів dirs (у порядку тек).

    Тека з session.json.enc (або лише з відео – перерваний запис) – живий запис;
    тека з підтеками <запис>_sNN/session.json.enc – імпортований запис, по сеансу
    на підтеку.
    """
    refs = []
    for d in dirs:
        subs = sorted(s for s in d.iterdir() if s.is_dir() and (s / ("session.json" + SUFFIX)).exists())
        if subs and not (d / ("session.json" + SUFFIX)).exists():
            refs.extend(SessionRef(s.name, s, d, "imported") for s in subs)
        else:
            refs.append(SessionRef(d.name, d, d, "live"))
    return refs


def video_file(folder: Path) -> Path | None:
    """Зашифроване відео запису: video.mp4.enc, video.avi.enc тощо."""
    found = sorted(folder.glob("video.*" + SUFFIX))
    return found[0] if found else None


# ----- декодування відео в пам'яті -----

class MemoryReader(io.BufferedIOBase):
    """Потік для cv2.VideoCapture над розшифрованим буфером (без копії і без файлу)."""

    def __init__(self, buf: bytearray):
        super().__init__()
        self._view = memoryview(buf)
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        end = len(self._view) if size is None or size < 0 else min(len(self._view), self._pos + size)
        data = bytes(self._view[self._pos:end])
        self._pos = end
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: len(self._view)}.get(whence, 0)
        self._pos = max(0, min(len(self._view), base + offset))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def close(self) -> None:
        self._view.release()
        super().close()


def encode_frame(frame: np.ndarray, width: int = REVIEW_WIDTH, quality: int = JPEG_QUALITY) -> bytes:
    """Зменшує кадр до width пікселів завширшки (якщо він ширший) і стискає в JPEG."""
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ReviewError("кадр не стиснуто в JPEG", 500)
    return jpeg.tobytes()


def _open(buf: bytearray) -> tuple[cv2.VideoCapture, MemoryReader]:
    reader = MemoryReader(buf)
    cap = cv2.VideoCapture(reader, cv2.CAP_FFMPEG, [])
    if not cap.isOpened():
        reader.close()
        raise ReviewError("відео запису не відкривається (потрібен OpenCV з FFmpeg)", 500)
    return cap, reader


def _close(cap: cv2.VideoCapture, reader: MemoryReader) -> None:
    try:
        cap.release()
    except cv2.error:  # деякі збірки OpenCV кидають виняток при закритті потоку з пам'яті
        pass
    reader.close()


def decode_indices(buf: bytearray, indices: list[int], width: int = REVIEW_WIDTH,
                   quality: int = JPEG_QUALITY) -> list[bytes]:
    """Кадри з номерами indices (за зростанням) з відео в буфері, як JPEG."""
    if not indices:
        return []
    wanted = set(indices)
    last = max(indices)
    cap, reader = _open(buf)
    out: dict[int, bytes] = {}
    try:
        i = 0
        first = min(indices)
        # MJPG складається лише з ключових кадрів, тому перехід точний; для mp4v перший кадр – 0
        if first > 0 and cap.set(cv2.CAP_PROP_POS_FRAMES, first):
            i = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            if i > first:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                i = 0
        while i <= last:
            if i in wanted:
                ok, frame = cap.read()
                if not ok:
                    break
                out[i] = encode_frame(frame, width, quality)
            elif not cap.grab():
                break
            i += 1
    finally:
        _close(cap, reader)
    return [out[i] for i in sorted(out)]


def decode_window(buf: bytearray, t_from: float, t_to: float, width: int = REVIEW_WIDTH,
                  quality: int = JPEG_QUALITY) -> tuple[list[bytes], list[float]]:
    """Кадри з мітками часу файлу в [t_from; t_to] (запис без frame_times.json)."""
    cap, reader = _open(buf)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, times = [], []
    try:
        i, prev = 0, -1.0
        while True:
            if not cap.grab():
                break
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            t = pos if pos > prev and pos < 1e5 else i / fps
            prev = t
            i += 1
            if t < t_from - 1e-6:
                continue
            if t > t_to + 1e-6:
                break
            ok, frame = cap.retrieve()
            if ok:
                frames.append(encode_frame(frame, width, quality))
                times.append(t)
    finally:
        _close(cap, reader)
    return frames, times


def thin(indices: list[int], limit: int = MAX_FRAMES) -> list[int]:
    """Рівномірно проріджує список, якщо він довший за limit."""
    if len(indices) <= limit:
        return indices
    pick = np.linspace(0, len(indices) - 1, limit).round().astype(int)
    return [indices[k] for k in sorted(set(pick.tolist()))]


# ----- метадані перегляду -----

def _num(v):
    return None if v is None or not isinstance(v, (int, float)) or not math.isfinite(v) else float(v)


def challenge_name(ch: dict) -> str:
    kind = ch.get("challenge_type") or ch.get("type")
    if kind == "turn_head":
        return "Поворот голови " + ("ліворуч" if ch.get("direction") == "left" else "праворуч")
    return CHALLENGE_NAMES.get(kind, kind or "?")


def challenge_short(ch: dict) -> str:
    kind = ch.get("challenge_type") or ch.get("type")
    if kind == "turn_head":
        return "поворот " + ("ліворуч" if ch.get("direction") == "left" else "праворуч")
    return CHALLENGE_SHORT.get(kind, kind or "?")


def model_scores(record: dict, classifier) -> dict[tuple[int, int], float]:
    """Оцінки поточної моделі для виконаних викликів: {(спроба, порядок): оцінка}.

    Ознаки обчислюються з рядів s_ref(t), s_enr(t) так само, як у build_dataset.py.
    """
    expl = record["decision"]["explanation"]
    st = record.get("settings", {})
    extractor = DriftFeatureExtractor(st.get("before_s", 1.0), st.get("after_s", 1.5),
                                      st.get("recovery_ratio", 0.9))
    t0 = expl.get("t0") or 0.0
    ser = expl["series"]
    t = np.array(ser["t"], dtype=float)
    nan = float("nan")
    s_ref = np.array([nan if v is None else v for v in ser["s_ref"]], dtype=float)
    s_enr = np.array([nan if v is None else v for v in ser["s_enr"]], dtype=float)
    keys, rows = [], []
    for a, challenges in enumerate(expl.get("attempts") or [], start=1):
        for k, ch in enumerate(challenges):
            if not ch.get("performed") or ch.get("t_start") is None or ch.get("t_end") is None:
                continue
            try:
                fv = extractor.extract_series(t, s_ref, s_enr, ch["t_start"] - t0, ch["t_end"] - t0)
            except InsufficientDataError:
                continue
            keys.append((a, k))
            rows.append(fv.values)
    if not rows:
        return {}
    scores = classifier.predict_batch(np.stack(rows))
    return {key: float(s) for key, s in zip(keys, scores)}


def trained(classifier) -> bool:
    """Чи завантажено навчену модель (а не базовий поріг до навчання)."""
    return getattr(classifier, "version", "untrained") not in ("threshold-default", "untrained")


def build_meta(record: dict, frame_times: list[float], *, uid: str, layout: str, classifier=None,
               threshold: float | None = None, frame_size: tuple[int, int] | None = None) -> dict:
    """Метадані перегляду з запису у форматі session.json."""
    meta = record.get("metadata", {})
    st = record.get("settings", {})
    dec = record.get("decision", {})
    expl = dec.get("explanation", {})
    t0 = expl.get("t0") or 0.0
    ser = expl.get("series") or {"t": [], "s_ref": [], "s_enr": []}
    thr = dec.get("threshold", st.get("threshold", threshold if threshold is not None else 0.5))

    offline = dec.get("decision") == "offline" or dec.get("decision") is None
    use_model = offline and classifier is not None and trained(classifier)
    try:
        mscores = model_scores(record, classifier) if use_model else {}
    except Exception:  # оцінка лише для перегляду
        log.exception("оцінку поточною моделлю не обчислено")
        mscores, use_model = {}, False

    challenges = []
    attempts = expl.get("attempts") or []
    for a, chs in enumerate(attempts, start=1):
        for k, ch in enumerate(chs):
            ts, te = _num(ch.get("t_start")), _num(ch.get("t_end"))
            challenges.append({
                "attempt": a, "order": k, "type": ch.get("challenge_type"),
                "direction": ch.get("direction"),
                "name": challenge_name(ch), "short": challenge_short(ch),
                "t_start": None if ts is None else ts - t0, "t_end": None if te is None else te - t0,
                "performed": bool(ch.get("performed")),
                "performed_at": None if _num(ch.get("performed_at")) is None else ch["performed_at"] - t0,
                "score": _num(ch.get("score")), "model_score": mscores.get((a, k)),
                "coverage": _num(ch.get("coverage")), "yaw_deg": _num(ch.get("yaw_deg")),
            })

    performed = sum(c["performed"] for c in challenges)
    decision = {"decision": dec.get("decision"), "score": _num(dec.get("score")), "threshold": thr,
                "is_attack": dec.get("is_attack"), "source": "session",
                "model_version": expl.get("model_version"), "note": "", "reason": expl.get("reason")}
    if decision["decision"] == "failed":
        decision["note"] = "не пройдено: " + FAILURE_TEXT.get(expl.get("reason"), "сеанс не завершено")
    if offline:
        decision.update(decision="offline", source="none", score=None,
                        note="оцінка не обчислювалася під час запису")
        if use_model:
            thr = threshold if threshold is not None else thr
            if mscores:
                score = session_score(list(mscores.values()))
                decision.update(decision="attack" if score >= thr else "bona_fide", score=score,
                                threshold=thr, is_attack=score >= thr, source="model",
                                model_version=getattr(classifier, "version", "?"),
                                note="оцінка поточною моделлю")
            else:
                decision.update(source="model", model_version=getattr(classifier, "version", "?"),
                                note="жоден виклик не виконано – оцінку поточною моделлю не обчислено")

    series_t = list(ser.get("t") or [])
    duration = expl.get("duration_s")
    if duration is None and series_t:
        duration = series_t[-1] - (series_t[0] or 0.0)
    end = max([*(frame_times[-1:] or [0.0]), *(series_t[-1:] or [0.0])])
    cond = meta.get("conditions") or {}
    return {
        "uid": uid, "layout": layout,
        "participant": meta.get("participant"), "presentation_type": meta.get("presentation_type"),
        "swap_tool": meta.get("swap_tool") or "", "target_face": meta.get("target_face") or "",
        "claimed_identity": meta.get("claimed_identity") or expl.get("claimed_person_id"),
        "session_index": meta.get("session_index"), "interface": meta.get("interface"),
        "source": meta.get("source") or cond.get("source"), "lighting": cond.get("lighting"),
        "recorded_at": meta.get("recorded_at"),
        "settings": {"before_s": st.get("before_s", 1.0), "after_s": st.get("after_s", 1.5), "threshold": thr},
        "duration_s": duration, "end_s": end,
        "series": {"t": series_t, "s_ref": list(ser.get("s_ref") or []), "s_enr": list(ser.get("s_enr") or [])},
        "challenges": challenges, "performed": performed, "count": len(challenges),
        "decision": decision,
        "frames": {"count": len(frame_times), "t": [round(float(v), 4) for v in frame_times],
                   "width": frame_size[0] if frame_size else None,
                   "height": frame_size[1] if frame_size else None},
    }


def _frame_size(jpeg: bytes | None) -> tuple[int, int] | None:
    if not jpeg:
        return None
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    return (img.shape[1], img.shape[0]) if img is not None else None


def prepare(ref: SessionRef, key: bytes, classifier=None, threshold: float | None = None) -> ReviewData:
    """Розшифровує анотацію і відео сеансу в пам'яті, декодує кадри сеансу."""
    try:
        record = read_json_encrypted(ref.annotation, key)
    except FileNotFoundError:
        raise ReviewError("сеанс перервано: анотації немає, показати графік і виклики неможливо") from None
    expl = record.get("decision", {}).get("explanation", {})
    series_t = (expl.get("series") or {}).get("t") or []
    video = video_file(ref.recording)
    frames: list[bytes] = []
    times: list[float] = []
    if video is None:
        log.warning("перегляд %s: відео не знайдено", ref.uid)
    else:
        buf = read_encrypted_buffer(video, key)
        try:
            if ref.layout == "live":
                idx = thin(list(range(len(series_t))))
                frames = decode_indices(buf, idx)
                times = [float(series_t[i]) for i in idx[:len(frames)]]
            else:
                t0 = expl.get("t0") or 0.0
                lo = t0 + (series_t[0] if series_t else 0.0)
                hi = t0 + (series_t[-1] if series_t else 0.0)
                ft = ref.recording / ("frame_times.json" + SUFFIX)
                if ft.exists():
                    abs_times = json.loads(read_encrypted(ft, key).decode("utf-8"))
                    idx = thin([i for i, v in enumerate(abs_times) if lo - 1e-6 <= v <= hi + 1e-6])
                    frames = decode_indices(buf, idx)
                    times = [float(abs_times[i]) - t0 for i in idx[:len(frames)]]
                else:
                    frames, file_times = decode_window(buf, lo, hi)
                    times = [v - t0 for v in file_times]
        finally:
            del buf
    meta = build_meta(record, times, uid=ref.uid, layout=ref.layout, classifier=classifier,
                      threshold=threshold, frame_size=_frame_size(frames[0] if frames else None))
    if video is None:
        meta["frames"]["missing"] = True
    return ReviewData(meta, frames)


__all__ = ["DecryptionError", "MemoryReader", "ReviewData", "ReviewError", "SessionRef", "UID_RE",
           "build_meta", "decode_indices", "decode_window", "encode_frame", "find_sessions", "prepare",
           "trained", "video_file"]
