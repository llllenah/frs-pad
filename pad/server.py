"""Веб-інтерфейс прототипу: бекенд FastAPI (REST + WebSocket) для екранних форм 4.7 і 4.8.

Браузер захоплює кадри вебкамери (getUserMedia), стискає їх у JPEG і надсилає
через WebSocket /ws/stream. Сервер обробляє кадр тим самим конвеєром, що й
scripts/demo.py (FramePipeline -> SessionManager), і повертає ключові точки,
стан сеансу, ряди s_ref(t), s_enr(t), ознаки, оцінку і рішення. Записи набору
даних (форма 4.8) зберігаються лише зашифрованими (pad/storage.py, FR-65) у
тому самому форматі, що й записи scripts/record_session.py. У режимі моніторингу
той самий потік кадрів іде на ідентифікацію 1:N (pad/identify.py): кожне обличчя
кадру порівнюється з еталонами всіх зареєстрованих осіб.

REST:
    GET  /api/config                     параметри модуля, типи викликів, стан ключа
    GET  /api/participants               псевдоніми із зареєстрованим еталоном
    POST /api/participants/{id}/image    еталон з фотографії (тіло запиту – JPEG або PNG)
    POST /api/sessions                   новий сеанс верифікації або запису
    GET  /api/sessions/{id}              стан сеансу
    POST /api/sessions/{id}/cancel       перервати сеанс
    GET  /api/recordings                 записані сеанси і підсумок набору даних
    POST /api/recordings/{name}/review   позначка «розмічено» або «очікує перевірки»
    GET  /api/stats                      FPS, час етапів і затримка рішень
    GET  /api/monitor                    база еталонів для моніторингу і журнал подій
WebSocket /ws/stream:
    текст {"cmd": "attach", "session_id": 1} – наступні кадри йдуть у сеанс;
    текст {"cmd": "enroll", "person_id": "P01"} – реєстрація еталона за 30 кадрами;
    текст {"cmd": "monitor", "on": true} – наступні кадри йдуть на ідентифікацію 1:N
        (база еталонів перечитується), {"cmd": "monitor", "on": false} – вимкнути;
    двійкове повідомлення – 8 байт часу кадру (float64, little-endian, с) і JPEG.
"""

from __future__ import annotations

import json
import logging
import math
import re
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .app import STATE_TEXT, RunStats, _json_default
from .classifiers import session_score
from .features import FEATURE_NAMES
from .identify import FaceGallery, MonitorSettings
from .pipeline import enroll_embedding
from .session import SessionManager, VerificationSession
from .settings import CHALLENGE_TYPES, PadSettings
from .storage import (SUFFIX, DecryptionError, encrypt_file, find_embedding, load_embedding,
                      read_json_encrypted, save_embedding, write_encrypted)
from .structures import SessionState

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"

#: Псевдонім учасника або цільового обличчя; входить в імена файлів, тому без «/», «.» тощо.
PSEUDONYM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")

#: Скільки кадрів збирається для еталона заявленої особи (як клавіша E у demo.py).
ENROLL_FRAMES = 30

#: Скільки сеансів тримати в пам'яті; старі завершені видаляються.
MAX_SESSIONS = 50

PROTOCOL = "DS-PILOT-01"

CHALLENGE_LABELS = {
    "cover_left": ("Закрити ліву половину обличчя долонею", "ЛП"),
    "cover_right": ("Закрити праву половину обличчя долонею", "ПП"),
    "cover_mouth": ("Закрити рот і підборіддя долонею", "Р"),
    "turn_head": ("Поворот голови вліво / вправо ≥ 30°", "ПГ"),
    "hand_swipe": ("Провести рукою перед обличчям", "ПР"),
}

SWAP_TOOLS = {"deep_live_cam": "Deep-Live-Cam", "deepfacelive": "DeepFaceLive", "other": "Інший"}

LIGHTING = {"normal": "Нормальне (≈300 лк)", "low": "Слабке (≈50 лк)",
            "bright": "Яскраве (≈1000 лк)", "backlight": "Контрове (світло ззаду)"}

DECISION_STATES = {"bona_fide", "attack", "failed"}

#: Скільки секунд після останнього кадру з невідомою особою тримати сповіщення (щоб не блимало).
ALERT_HOLD_S = 2.0

#: Скільки останніх подій журналу моніторингу надсилати з кожним кадром.
EVENTS_IN_FRAME = 12


# ----- запити REST -----

class Scenario(BaseModel):
    """Сценарій викликів сеансу запису (набір типів, кількість, тривалість)."""

    challenge_types: list[str] = Field(default_factory=lambda: list(CHALLENGE_TYPES))
    challenge_count: int = 3
    duration_s: float = 5.0


class RecordingMeta(BaseModel):
    """Параметри сеансу запису з форми 4.8."""

    participant: str
    consent: bool = False
    consent_ref: str = ""
    presentation_type: Literal["bona_fide", "attack"]
    swap_tool: str = ""
    target_face: str = ""
    lighting: str = "normal"
    webcam: str = ""
    source: str = ""
    scenario: Scenario = Field(default_factory=Scenario)


class SessionRequest(BaseModel):
    """Новий сеанс: верифікація (форма 4.7) або запис для набору даних (форма 4.8)."""

    claimed_person_id: str = ""
    operator_id: str = "operator"
    mode: Literal["verify", "record"] = "verify"
    origin: Literal["", "monitor"] = ""   # «monitor» – перевірку запущено з вкладки «Моніторинг»
    recording: RecordingMeta | None = None


class ReviewRequest(BaseModel):
    """Розмітка запису: оператор перевірив виклики і мітки сеансу."""

    reviewed: bool = True
    note: str = ""


# ----- запис сеансу -----

class SessionRecorder:
    """Пише кадри сеансу у video.mp4, після завершення шифрує відео і анотацію.

    Браузер надсилає кадри з нерівним інтервалом, тому частота кадрів відео
    оцінюється за мітками перших FPS_PROBE кадрів. Як і в DemoApp, кадр i
    відео відповідає мітці series.t[i] в анотації.
    """

    FPS_PROBE = 15

    def __init__(self, folder: Path, key: bytes):
        self.folder = folder
        self.key = key
        self.writer: cv2.VideoWriter | None = None
        self.size: tuple[int, int] | None = None
        self.fps: float | None = None
        self._buffer: list[tuple[float, np.ndarray]] = []

    def write(self, frame: np.ndarray, t: float) -> None:
        if self.writer is None:
            self._buffer.append((t, frame))
            if len(self._buffer) >= self.FPS_PROBE:
                self._open()
            return
        self._write(frame)

    def _write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self.writer.write(frame)

    def _open(self) -> None:
        ts = [t for t, _ in self._buffer]
        fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 30.0
        self.fps = float(np.clip(fps, 1.0, 60.0))
        h, w = self._buffer[0][1].shape[:2]
        self.size = (w, h)
        self.writer = cv2.VideoWriter(str(self.folder / "video.mp4"),
                                      cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
        for _, frame in self._buffer:
            self._write(frame)
        self._buffer = []

    def _close_video(self) -> None:
        if self.writer is None and self._buffer:
            self._open()
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        video = self.folder / "video.mp4"
        if video.exists():
            encrypt_file(video, self.key)

    def finish(self, record: dict) -> None:
        """Шифрує відео і записує анотацію session.json.enc."""
        self._close_video()
        data = json.dumps(record, ensure_ascii=False, indent=1, default=_json_default).encode("utf-8")
        write_encrypted(self.folder / "session.json", data, self.key)

    def abort(self) -> None:
        """Сеанс перервано: відео шифрується, анотація не створюється (як у DemoApp)."""
        self._close_video()


# ----- стан сервера -----

@dataclass
class LiveSession:
    """Сеанс, яким керує веб-інтерфейс."""

    manager: SessionManager
    session: VerificationSession
    mode: str
    code: str
    enrolled_source: str | None = None
    metadata: dict = field(default_factory=dict)
    recorder: SessionRecorder | None = None
    record_name: str | None = None
    cancelled: bool = False
    decision: dict | None = None
    final_frame_ms: float | None = None
    origin: str = ""


@dataclass
class StreamConnection:
    """Стан одного з'єднання WebSocket: прив'язаний сеанс, реєстрація еталона або моніторинг."""

    session_id: int | None = None
    enroll_id: str | None = None
    enroll_frames: list = field(default_factory=list)
    monitor: bool = False
    # моніторинг: для кожної особи (або «unknown») – кадрів поспіль, час останнього кадру, чи є подія
    presence: dict = field(default_factory=dict)


def check_pseudonym(value: str, what: str) -> str:
    value = (value or "").strip()
    if not PSEUDONYM_RE.match(value):
        raise HTTPException(422, f"{what}: лише псевдонім з латинських літер, цифр, «-» і «_» "
                                 f"(наприклад, P01), до 32 символів")
    return value


def challenge_code(ch: dict) -> str:
    """Скорочення виклику для таблиці записів: ЛП, ПП, Р, ПГ-Л, ПГ-П, ПР."""
    kind = ch.get("challenge_type") or ch.get("type")
    short = CHALLENGE_LABELS.get(kind, (kind, kind))[1]
    if kind == "turn_head" and ch.get("direction"):
        short += "-Л" if ch["direction"] == "left" else "-П"
    return short


def needs_hands(s: VerificationSession) -> bool:
    """Hands потрібен лише під час викликів з кистю (як DemoApp._needs_hands)."""
    if s.state not in (SessionState.GENERATING, SessionState.CHALLENGE):
        return False
    ch = s.current if s.state == SessionState.CHALLENGE else None
    return ch is None or ch.type != "turn_head"


def jsonable(obj):
    """Готує відповідь до JSON: numpy -> Python, NaN і нескінченності -> None."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, SessionState):
        return obj.value
    return obj


def _r(x, digits: int = 4):
    return None if x is None or not math.isfinite(x) else round(float(x), digits)


def _stamp_of(name: str) -> str:
    """Час запису з імені теки <псевдонім>_<тип>_<РРРРММДД>_<ГГХХСС>[_n]."""
    m = re.search(r"_(\d{8}_\d{6})(?:_\d+)?$", name)
    return m.group(1) if m else ""


class PadService:
    """Спільний стан веб-інтерфейсу: конвеєр, сеанси, еталони, записи."""

    def __init__(self, pipeline, manager: SessionManager, data_dir: str | Path,
                 key: bytes | None = None, monitoring: dict | MonitorSettings | None = None):
        self.pipeline = pipeline
        self.manager = manager
        self.data_dir = Path(data_dir)
        self.enrolled_dir = self.data_dir / "enrolled"
        self.recordings_dir = self.data_dir / "recordings"
        self.key = key
        self.memory_enrolled: dict[str, np.ndarray] = {}
        self.sessions: dict[int, LiveSession] = {}
        self.stats = RunStats()
        self.monitor = (monitoring if isinstance(monitoring, MonitorSettings)
                        else MonitorSettings.from_dict(monitoring))
        self.gallery = FaceGallery(self.monitor.threshold)
        self.gallery_locked: list[str] = []
        self.events: deque = deque(maxlen=self.monitor.max_events)
        # MediaPipe і SessionManager не розраховані на паралельні виклики: кадри обробляються по одному
        self.lock = threading.RLock()

    # ----- довідкові дані -----

    def config(self) -> dict:
        clf = self.manager.classifier
        return {
            "settings": self.manager.settings.to_dict(),
            "challenge_types": [{"id": k, "label": CHALLENGE_LABELS[k][0], "short": CHALLENGE_LABELS[k][1]}
                                for k in CHALLENGE_TYPES],
            "feature_names": FEATURE_NAMES,
            "detector": getattr(self.pipeline.detector, "name", "?"),
            "embedder": getattr(self.pipeline.embedder, "name", "?"),
            "classifier": getattr(clf, "version", "unknown"),
            "encryption": {"algorithm": "AES-256-GCM", "key_set": self.key is not None},
            "protocol": PROTOCOL,
            "swap_tools": SWAP_TOOLS,
            "lighting": LIGHTING,
            "enroll_frames": ENROLL_FRAMES,
            "monitoring": {**self.monitor.to_dict(), "backend": self.gallery.backend},
        }

    def participants(self) -> list[dict]:
        found: dict[str, str] = {pid: "memory" for pid in self.memory_enrolled}
        if self.enrolled_dir.exists():
            for path in sorted(self.enrolled_dir.iterdir()):
                name = path.name
                if name.endswith(".npy" + SUFFIX):
                    found[name[:-len(".npy" + SUFFIX)]] = "encrypted"
                elif name.endswith(".npy"):
                    found.setdefault(name[:-4], "plain")
        return [{"id": pid, "stored": found[pid]} for pid in sorted(found) if PSEUDONYM_RE.match(pid)]

    def load_enrolled(self, person_id: str) -> tuple[np.ndarray | None, str | None]:
        """Еталон заявленої особи: з пам'яті (реєстрація без ключа) або з data/enrolled."""
        if person_id in self.memory_enrolled:
            return self.memory_enrolled[person_id], "memory"
        path = find_embedding(self.enrolled_dir / f"{person_id}.npy")
        if path is None:
            return None, None
        if path.name.endswith(SUFFIX) and self.key is None:
            log.warning("еталон %s зашифровано, а PAD_DATASET_KEY не задано", person_id)
            return None, "locked"
        try:
            return load_embedding(path, self.key), "encrypted" if path.name.endswith(SUFFIX) else "plain"
        except DecryptionError as exc:
            log.warning("еталон %s не розшифровано: %s", person_id, exc)
            return None, "locked"

    def store_embedding(self, person_id: str, emb: np.ndarray) -> str:
        """Зберігає еталон зашифрованим (якщо задано ключ), інакше лише в пам'яті."""
        self.memory_enrolled[person_id] = emb
        self.gallery.add(person_id, emb)
        if self.key is None:
            return "memory"
        self.enrolled_dir.mkdir(parents=True, exist_ok=True)
        save_embedding(self.enrolled_dir / f"{person_id}.npy", emb, self.key)
        return "encrypted"

    def enroll_image(self, person_id: str, body: bytes) -> dict:
        person_id = check_pseudonym(person_id, "псевдонім")
        img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR) if body else None
        if img is None:
            raise HTTPException(422, "очікується зображення JPEG або PNG")
        with self.lock:
            emb = self.pipeline.embed_image(img)
            if emb is None:
                raise HTTPException(422, "на фотографії обличчя не виявлено")
            stored = self.store_embedding(person_id, emb)
        return {"person_id": person_id, "stored": stored}

    # ----- сеанси -----

    def create_session(self, req: SessionRequest) -> dict:
        manager = self.manager
        metadata: dict = {}
        if req.mode == "record":
            if req.recording is None:
                raise HTTPException(422, "для запису потрібні параметри сеансу (recording)")
            rec = req.recording
            participant = check_pseudonym(rec.participant, "учасник")
            if not rec.consent:
                raise HTTPException(422, "запис без письмової згоди учасника заборонено")
            if self.key is None:
                raise HTTPException(409, "записи зберігаються лише зашифрованими: задайте ключ у "
                                         "PAD_DATASET_KEY і перезапустіть сервер")
            attack = rec.presentation_type == "attack"
            if attack and not rec.swap_tool.strip():
                raise HTTPException(422, "для атаки вкажіть інструмент підміни")
            target = check_pseudonym(rec.target_face, "цільове обличчя") if rec.target_face.strip() else ""
            claimed = req.claimed_person_id.strip() or (target if attack and target else participant)
            sc = rec.scenario
            try:
                settings = PadSettings.from_dict({**self.manager.settings.to_dict(),
                                                  "challenge_types": list(dict.fromkeys(sc.challenge_types)),
                                                  "challenge_count": sc.challenge_count,
                                                  "duration_s": sc.duration_s})
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, f"сценарій викликів: {exc}") from exc
            # окремий SessionManager зі сценарієм сеансу; лічильник ідентифікаторів спільний
            manager = SessionManager(self.manager.classifier, settings, verifier=self.manager.verifier,
                                     features=self.manager.features)
            manager._ids = self.manager._ids
            metadata = {
                "participant": participant, "consent": True, "consent_ref": rec.consent_ref.strip(),
                "presentation_type": rec.presentation_type,
                "swap_tool": rec.swap_tool.strip() if attack else "",
                "target_face": target, "claimed_identity": claimed,
                "conditions": {"lighting": rec.lighting, "webcam": rec.webcam.strip(),
                               "source": rec.source.strip() or "browser", "resolution": None},
                "scenario": {"challenge_types": settings.challenge_types,
                             "challenge_count": settings.challenge_count, "duration_s": settings.duration_s},
                "operator": req.operator_id, "interface": "web",
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
            }
        claimed = check_pseudonym(metadata.get("claimed_identity") or req.claimed_person_id,
                                  "заявлена особа")

        with self.lock:
            enrolled, source = self.load_enrolled(claimed)
            s = manager.start_session(claimed, req.operator_id, enrolled)
            code = f"{'VS' if req.mode == 'verify' else 'RS'}-{datetime.now():%Y-%m-%d}-{s.id:04d}"
            live = LiveSession(manager, s, req.mode, code, source, metadata, origin=req.origin)
            if req.mode == "record":
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"{metadata['participant']}_{metadata['presentation_type']}_{stamp}"
                folder, n = self.recordings_dir / name, 1
                while folder.exists():
                    n += 1
                    folder = self.recordings_dir / f"{name}_{n}"
                folder.mkdir(parents=True)
                live.recorder = SessionRecorder(folder, self.key)
                live.record_name = folder.name
            self.sessions[s.id] = live
            self._prune()
            log.info("сеанс %s (%s) для %s, еталон: %s", code, req.mode, claimed, source)
            return self.snapshot(live)

    def get(self, session_id: int) -> LiveSession:
        live = self.sessions.get(session_id)
        if live is None:
            raise HTTPException(404, f"сеанс {session_id} не знайдено")
        return live

    def cancel(self, session_id: int) -> dict:
        with self.lock:
            live = self.get(session_id)
            self._cancel(live)
            return self.snapshot(live)

    def _cancel(self, live: LiveSession) -> None:
        if live.session.state.is_final or live.cancelled:
            return
        live.cancelled = True
        if live.recorder is not None:
            live.recorder.abort()
            live.recorder = None
            log.warning("сеанс %s перервано, зашифровано лише відео", live.code)

    def _prune(self) -> None:
        done = [sid for sid, lv in self.sessions.items() if lv.cancelled or lv.session.state.is_final]
        for sid in done[: max(0, len(self.sessions) - MAX_SESSIONS)]:
            del self.sessions[sid]

    def _finish(self, live: LiveSession) -> None:
        """Рішення сформовано: затримка, журнал швидкодії, запис анотації (як DemoApp)."""
        s = live.session
        dec = s.decision.to_dict()
        t_last_end = max((c["t_end"] for a in dec["explanation"]["attempts"] for c in a
                          if c.get("t_end") is not None), default=None)
        latency = {
            "decision_compute_ms": s.decision_latency_ms,
            "final_frame_ms": live.final_frame_ms,
            "after_last_challenge_s": (s.last_t - t_last_end) if t_last_end is not None else None,
        }
        dec["explanation"]["latency"] = latency
        live.decision = dec
        self.stats.decisions.append({"session_id": s.id, "decision": dec["decision"], "score": dec["score"],
                                     **latency, "duration_s": dec["explanation"]["duration_s"]})
        log.info("сеанс %s: %s, оцінка %s", live.code, dec["decision"], dec["score"])
        if live.origin == "monitor":
            self._event("pad", s.claimed_person_id, None, decision=dec["decision"], score=dec["score"],
                        code=live.code)
        if live.recorder is not None:
            record = {"metadata": live.metadata, "settings": live.manager.settings.to_dict(),
                      "embedder": getattr(self.pipeline.embedder, "name", "?"), "decision": dec,
                      "annotation": {"reviewed": False}}
            live.recorder.finish(record)
            live.recorder = None
            log.info("запис збережено (AES-256-GCM): %s", live.record_name)

    def snapshot(self, live: LiveSession) -> dict:
        """Стан сеансу для екранних форм."""
        s, m = live.session, live.manager
        st = m.settings
        status = m.status(s)
        t0 = s.t_started or 0.0
        t, s_ref, s_enr = s.series_arrays()
        state = "cancelled" if live.cancelled else s.state.value

        done = [c for c in s.challenges if c.features is not None]
        scores: list[float] = []
        if done:
            try:
                scores = [float(x) for x in m.classifier.predict_batch(np.stack([c.features.values for c in done]))]
            except Exception:  # оцінка лише для відображення, сеанс від неї не залежить
                log.exception("попередня оцінка не обчислена")
        challenges = []
        for i, c in enumerate(s.challenges):
            if c.t_end is not None:
                cstate = "done" if c.performed_at is not None else "failed"
            elif i == s.index and s.state == SessionState.CHALLENGE and not live.cancelled:
                cstate = "current"
            else:
                cstate = "pending"
            score = c.score if c.score is not None else (scores[i] if i < len(scores) else None)
            challenges.append({"type": c.type, "instruction": c.instruction(),
                               "short": challenge_code({"type": c.type, "direction": getattr(c, "direction", None)}),
                               "state": cstate, "score": score, "progress": c.progress()})

        cur = s.current if s.state in (SessionState.CHALLENGE, SessionState.FEATURES) else None
        measure = dict(cur.measures[-1]) if cur is not None and cur.measures else None
        if measure is not None:
            measure["type"] = cur.type

        spans = []
        if s.decision is None:
            spans += [(c["t_start"] - t0, c["t_end"] - t0) for a in s.attempts for c in a
                      if c.get("t_start") is not None and c.get("t_end") is not None]
            spans += [(c.t_start - t0, (c.t_end if c.t_end is not None else s.last_t) - t0)
                      for c in s.challenges if c.t_start is not None]
        else:
            spans = [(c["t_start"] - t0, c["t_end"] - t0) for a in s.attempts for c in a
                     if c.get("t_start") is not None and c.get("t_end") is not None]

        decision = None
        if live.decision is not None:
            expl = {k: v for k, v in live.decision["explanation"].items() if k != "series"}
            decision = {**{k: v for k, v in live.decision.items() if k != "explanation"}, "explanation": expl}

        return jsonable({
            "id": s.id, "code": live.code, "mode": live.mode, "origin": live.origin, "state": state,
            "state_text": STATE_TEXT.get(state, "перервано" if state == "cancelled" else state),
            "claimed_person_id": s.claimed_person_id, "enrolled": live.enrolled_source,
            "status": status, "challenge_duration_s": cur.duration_s if cur is not None else st.duration_s,
            "challenges": challenges, "measure": measure,
            "features": done[-1].features.as_dict() if done else None,
            "features_challenge": done[-1].instruction() if done else None,
            "preliminary": ({"score": session_score(np.array(scores)), "n": len(scores),
                             "count": len(s.challenges)} if scores and live.decision is None else None),
            "series": {"t": [_r(v, 3) for v in t - t0], "s_ref": [_r(v) for v in s_ref],
                       "s_enr": [_r(v) for v in s_enr]},
            "spans": spans,
            "settings": {"threshold": st.threshold, "before_s": st.before_s, "after_s": st.after_s,
                         "min_coverage": st.min_coverage, "min_yaw_deg": st.min_yaw_deg,
                         "min_shift": st.min_shift},
            "decision": decision,
            "record_name": live.record_name,
        })

    # ----- кадри і команди WebSocket -----

    def command(self, conn: StreamConnection, msg: dict) -> dict:
        cmd = msg.get("cmd")
        with self.lock:
            if cmd == "attach":
                try:
                    live = self.get(int(msg.get("session_id")))
                except (TypeError, ValueError, HTTPException):
                    return {"type": "error", "message": "сеанс не знайдено"}
                conn.session_id = live.session.id
                conn.enroll_id = None
                conn.monitor = False
                return {"type": "attached", "session": self.snapshot(live)}
            if cmd == "detach":
                conn.session_id = None
                return {"type": "detached"}
            if cmd == "enroll":
                try:
                    pid = check_pseudonym(str(msg.get("person_id", "")), "псевдонім")
                except HTTPException as exc:
                    return {"type": "error", "message": exc.detail}
                live = self.sessions.get(conn.session_id) if conn.session_id else None
                if live is not None and not live.cancelled and not live.session.state.is_final:
                    return {"type": "error", "message": "під час сеансу реєстрація недоступна"}
                conn.session_id = None
                conn.monitor = False
                conn.enroll_id, conn.enroll_frames = pid, []
                return {"type": "enroll_started", "person_id": pid, "frames": ENROLL_FRAMES}
            if cmd == "monitor":
                if not msg.get("on", True):
                    conn.monitor = False
                    return {"type": "monitor_stopped"}
                live = self.sessions.get(conn.session_id) if conn.session_id else None
                if live is not None and not live.cancelled and not live.session.state.is_final:
                    return {"type": "error", "message": "під час сеансу верифікації моніторинг недоступний"}
                conn.session_id = conn.enroll_id = None
                conn.enroll_frames, conn.presence = [], {}
                conn.monitor = True
                self.reload_gallery()
                return {"type": "monitor_started", "monitor": self.monitor_status()}
        return {"type": "error", "message": f"невідома команда {cmd!r}"}

    def disconnect(self, conn: StreamConnection) -> None:
        """Браузер закрито посеред сеансу: сеанс переривається, відео шифрується."""
        with self.lock:
            live = self.sessions.get(conn.session_id) if conn.session_id else None
            if live is not None:
                self._cancel(live)

    def process_frame(self, conn: StreamConnection, t: float, payload: bytes) -> dict:
        frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR) if payload else None
        if frame is None:
            return {"type": "frame_error", "message": "кадр не розпізнано (очікується JPEG)"}
        if conn.monitor:
            with self.lock:
                return jsonable(self._monitor_frame(conn, t, frame))
        with self.lock:
            live = self.sessions.get(conn.session_id) if conn.session_id else None
            active = live is not None and not live.cancelled and not live.session.state.is_final
            data, timings = self.pipeline.process(frame, t, landmarks=True,
                                                  hands=active and needs_hands(live.session))
            session_ms = None
            if active:
                if live.recorder is not None:
                    live.recorder.write(frame, t)
                    live.metadata["conditions"]["resolution"] = f"{frame.shape[1]}x{frame.shape[0]}"
                tp = time.perf_counter()
                live.manager.on_frame(live.session.id, data)
                session_ms = (time.perf_counter() - tp) * 1000
                if live.session.state.is_final:
                    live.final_frame_ms = session_ms
                    self._finish(live)
            self.stats.add(timings, session_ms)
            msg = {"type": "frame", "t": t, "size": list(data.size),
                   "face": None if data.face is None else np.round(data.face).astype(int).ravel(),
                   "hand": None if data.hand is None else np.round(data.hand).astype(int).ravel(),
                   "box": None, "timings": {k: round(v, 1) for k, v in timings.items()},
                   "fps": round(self.stats.fps(), 1),
                   "pad_ms": None if session_ms is None else round(timings["landmarks_ms"] + session_ms, 1),
                   "check": self._check(data)}
            box = self.pipeline.last_box
            if box is not None:
                msg["box"] = [*(float(v) for v in box.bbox), float(box.score)]
            if conn.enroll_id is not None:
                msg["enroll"] = self._enroll_step(conn, data)
            if live is not None:
                msg["session"] = self.snapshot(live)
        return jsonable(msg)

    def _check(self, data) -> dict:
        """Блок «Перевірка виконання»: кількість точок і кути голови."""
        yaw = pitch = None
        if data.face is not None:
            try:
                yaw, pitch, _ = self.pipeline.verifier.head_pose(data.face, data.size)
            except Exception:  # кути лише для відображення
                pass
        return {"face_points": 0 if data.face is None else len(data.face),
                "hand_points": 0 if data.hand is None else len(data.hand),
                "yaw": yaw, "pitch": pitch, "det_score": data.det_score,
                "embedding": data.embedding is not None}

    def _enroll_step(self, conn: StreamConnection, data) -> dict:
        pid = conn.enroll_id
        conn.enroll_frames.append(data)
        n = len(conn.enroll_frames)
        if n < ENROLL_FRAMES:
            return {"person_id": pid, "progress": n / ENROLL_FRAMES, "done": False}
        emb = enroll_embedding(conn.enroll_frames, self.manager.settings.min_det_score)
        conn.enroll_id, conn.enroll_frames = None, []
        if emb is None:
            return {"person_id": pid, "progress": 1.0, "done": True, "ok": False,
                    "message": "обличчя не виявлено з потрібною упевненістю, повторіть реєстрацію"}
        stored = self.store_embedding(pid, emb)
        log.info("еталон %s зареєстровано (%s)", pid, stored)
        return {"person_id": pid, "progress": 1.0, "done": True, "ok": True, "stored": stored}

    # ----- моніторинг: ідентифікація 1:N -----

    def reload_gallery(self) -> None:
        """Перечитує еталони всіх осіб: з пам'яті сервера і з data/enrolled (з ключем)."""
        items, locked = {}, []
        for p in self.participants():
            emb, _ = self.load_enrolled(p["id"])
            if emb is None:
                locked.append(p["id"])
            else:
                items[p["id"]] = emb
        self.gallery.replace(items)
        self.gallery_locked = locked
        log.info("моніторинг: еталонів %d, не розшифровано %d (%s)", len(self.gallery), len(locked),
                 self.gallery.backend)

    def monitor_status(self) -> dict:
        """База еталонів, поріг і журнал подій для вкладки «Моніторинг»."""
        return jsonable({
            "threshold": self.gallery.threshold, "backend": self.gallery.backend,
            "ids": self.gallery.ids, "locked": self.gallery_locked,
            "key_set": self.key is not None, "settings": self.monitor.to_dict(),
            "events": list(self.events)[::-1],
        })

    def _event(self, kind: str, person_id: str | None, similarity: float | None, **extra) -> dict:
        ev = {"time": datetime.now().strftime("%H:%M:%S"), "kind": kind, "person_id": person_id,
              "similarity": similarity, **extra}
        self.events.append(ev)
        return ev

    def _monitor_frame(self, conn: StreamConnection, t: float, frame: np.ndarray) -> dict:
        """Кадр моніторингу: усі обличчя, найближчий еталон кожного, журнал і сповіщення.

        Подія «особа з'явилася» потрапляє в журнал, коли особу (або невідоме обличчя)
        розпізнано confirm_frames кадрів поспіль: окремий кадр з розмитим обличчям
        не дає хибного сповіщення. Поки особа лишається в кадрі (з пропусками не
        довшими за event_gap_s), повторна подія не пишеться.
        """
        st = self.monitor
        found, timings = self.pipeline.faces(frame, st.min_det_score, st.max_faces)
        tm = time.perf_counter()
        matches = self.gallery.search(np.stack([e for _, e in found])) if found else []
        timings["search_ms"] = (time.perf_counter() - tm) * 1000
        timings["total_ms"] += timings["search_ms"]
        self.stats.add(timings, None)

        faces, seen = [], {}
        for (box, _), m in zip(found, matches):
            faces.append({"box": [float(v) for v in box.bbox], "det_score": float(box.score),
                          "person_id": m.person_id, "similarity": m.similarity, "known": m.known,
                          "nearest": m.nearest})
            key = m.person_id or "unknown"
            sim = m.similarity if math.isfinite(m.similarity) else None
            if key not in seen or (sim is not None and (seen[key] is None or sim > seen[key])):
                seen[key] = sim

        for key, entry in list(conn.presence.items()):
            if key not in seen:
                entry["streak"] = 0
                if t - entry["last"] > st.event_gap_s:
                    del conn.presence[key]
        for key, sim in seen.items():
            entry = conn.presence.get(key)
            if entry is None or t - entry["last"] > st.event_gap_s:
                entry = conn.presence[key] = {"streak": 0, "last": t, "logged": False}
            entry["streak"] += 1
            entry["last"] = t
            if entry["streak"] >= st.confirm_frames and not entry["logged"]:
                entry["logged"] = True
                known = key != "unknown"
                self._event("known" if known else "unknown", key if known else None, sim)
                if not known:
                    log.warning("моніторинг: невідома особа, подібність %s", sim)

        unknown = conn.presence.get("unknown")
        alert = None
        if unknown is not None:
            if "unknown" in seen:
                unknown["count"] = sum(not f["known"] for f in faces)
                unknown["similarity"] = seen["unknown"]
            if unknown["logged"] and t - unknown["last"] <= ALERT_HOLD_S:
                alert = {"count": unknown["count"], "similarity": unknown["similarity"]}
        h, w = frame.shape[:2]
        return {"type": "frame", "t": t, "size": [w, h], "face": None, "hand": None, "box": None,
                "timings": {k: round(v, 1) for k, v in timings.items()},
                "fps": round(self.stats.fps(), 1), "pad_ms": None,
                "check": {"face_points": 0, "hand_points": 0, "yaw": None, "pitch": None,
                          "det_score": faces[0]["det_score"] if faces else 0.0, "embedding": bool(faces)},
                "monitor": {"faces": faces, "alert": alert, "threshold": self.gallery.threshold,
                            "gallery": len(self.gallery),
                            "events": list(self.events)[-EVENTS_IN_FRAME:][::-1]}}

    # ----- записи набору даних -----

    def _recording_dirs(self) -> list[Path]:
        if not self.recordings_dir.exists():
            return []
        dirs = [d for d in self.recordings_dir.iterdir() if d.is_dir()]
        return sorted(dirs, key=lambda d: (_stamp_of(d.name) or f"{d.stat().st_mtime:020.3f}", d.name))

    def recordings(self) -> dict:
        if self.key is None:
            raise HTTPException(409, "записи зашифровано: задайте PAD_DATASET_KEY і перезапустіть сервер")
        rows = []
        for i, d in enumerate(self._recording_dirs(), start=1):
            row = {"name": d.name, "id": f"S-{i:04d}", "participant": d.name.split("_")[0]}
            enc = d / ("session.json" + SUFFIX)
            if not enc.exists():
                row["status"] = "incomplete"   # сеанс перервано: є лише video.mp4.enc
                rows.append(row)
                continue
            try:
                rec = read_json_encrypted(enc, self.key)
            except DecryptionError:
                row["status"] = "locked"
                rows.append(row)
                continue
            meta, dec = rec.get("metadata", {}), rec.get("decision", {})
            expl = dec.get("explanation", {})
            attempts = expl.get("attempts") or [[]]
            cond = meta.get("conditions", {})
            reviewed = bool(rec.get("annotation", {}).get("reviewed"))
            row.update({
                "participant": meta.get("participant", row["participant"]),
                "presentation_type": meta.get("presentation_type"),
                "swap_tool": meta.get("swap_tool", ""), "target_face": meta.get("target_face", ""),
                "claimed_identity": meta.get("claimed_identity"),
                "challenges": [challenge_code(c) for c in attempts[-1]],
                "attempts": len(expl.get("attempts") or []),
                "lighting": cond.get("lighting"), "resolution": cond.get("resolution"),
                "webcam": cond.get("webcam"),
                "duration_s": expl.get("duration_s"), "decision": dec.get("decision"),
                "score": dec.get("score"),
                "status": "failed" if dec.get("decision") == "failed" else ("reviewed" if reviewed else "pending"),
                "note": rec.get("annotation", {}).get("note", ""),
                "interface": meta.get("interface", "cli"),
            })
            rows.append(row)
        complete = [r for r in rows if r.get("presentation_type")]
        summary = {
            "participants": len({r["participant"] for r in complete}),
            "sessions": len(complete),
            "bona_fide": sum(r["presentation_type"] == "bona_fide" for r in complete),
            "attack": sum(r["presentation_type"] == "attack" for r in complete),
            "tools": len({r["swap_tool"] for r in complete if r["presentation_type"] == "attack" and r["swap_tool"]}),
            "targets": len({r["target_face"] for r in complete if r["target_face"]}),
            "reviewed": sum(r["status"] == "reviewed" for r in complete),
        }
        return jsonable({"rows": rows[::-1], "summary": summary})

    def review(self, name: str, req: ReviewRequest) -> dict:
        if self.key is None:
            raise HTTPException(409, "записи зашифровано: задайте PAD_DATASET_KEY")
        folder = next((d for d in self._recording_dirs() if d.name == name), None)
        if folder is None:
            raise HTTPException(404, f"запис {name} не знайдено")
        enc = folder / ("session.json" + SUFFIX)
        if not enc.exists():
            raise HTTPException(409, "сеанс не завершено, анотації немає")
        with self.lock:
            rec = read_json_encrypted(enc, self.key)
            rec["annotation"] = {"reviewed": req.reviewed, "note": req.note.strip(),
                                 "reviewed_at": datetime.now().isoformat(timespec="seconds")}
            data = json.dumps(rec, ensure_ascii=False, indent=1, default=_json_default).encode("utf-8")
            write_encrypted(folder / "session.json", data, self.key)
        return {"name": name, "status": "reviewed" if req.reviewed else "pending"}


# ----- застосунок FastAPI -----

def create_app(service: PadService) -> FastAPI:
    """Застосунок FastAPI з REST, WebSocket і односторінковим інтерфейсом з pad/web."""
    app = FastAPI(title="FRS PAD", description="Прототип модуля виявлення атак підміни обличчя",
                  version="0.1.0")
    app.state.service = service
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/config")
    def get_config():
        return service.config()

    @app.get("/api/participants")
    def get_participants():
        return service.participants()

    @app.post("/api/participants/{person_id}/image")
    async def post_participant_image(person_id: str, request: Request):
        body = await request.body()
        return await run_in_threadpool(service.enroll_image, person_id, body)

    @app.post("/api/sessions")
    def post_session(req: SessionRequest):
        return service.create_session(req)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: int):
        with service.lock:
            return service.snapshot(service.get(session_id))

    @app.post("/api/sessions/{session_id}/cancel")
    def cancel_session(session_id: int):
        return service.cancel(session_id)

    @app.get("/api/recordings")
    def get_recordings():
        return service.recordings()

    @app.post("/api/recordings/{name}/review")
    def review_recording(name: str, req: ReviewRequest):
        return service.review(name, req)

    @app.get("/api/stats")
    def get_stats():
        with service.lock:
            return jsonable(service.stats.summary()) if service.stats.frames else {"frames": 0}

    @app.get("/api/monitor")
    def get_monitor():
        with service.lock:
            return service.monitor_status()

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        conn = StreamConnection()
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    data = msg["bytes"]
                    if len(data) <= 8:
                        reply = {"type": "frame_error", "message": "кадр без даних"}
                    else:
                        (t,) = struct.unpack_from("<d", data)
                        reply = await run_in_threadpool(service.process_frame, conn, t, data[8:])
                elif msg.get("text") is not None:
                    try:
                        cmd = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        cmd = {}
                    reply = await run_in_threadpool(service.command, conn, cmd if isinstance(cmd, dict) else {})
                else:
                    continue
                await ws.send_text(json.dumps(reply, ensure_ascii=False, allow_nan=False))
        except WebSocketDisconnect:
            pass
        finally:
            await run_in_threadpool(service.disconnect, conn)

    return app
