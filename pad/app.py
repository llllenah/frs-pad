"""Цикл демонстрації та запису сеансів: камера -> конвеєр -> SessionManager -> вікно.

Той самий цикл використовують scripts/demo.py (демонстрація і вимірювання
FPS та затримки) і scripts/record_session.py (запис пілотного набору даних).
Записи набору і ембединги учасників зберігаються лише зашифрованими
(pad/storage.py, FR-65).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .pipeline import FramePipeline, enroll_embedding
from .session import SessionManager
from .storage import encrypt_file, save_embedding, write_encrypted
from .structures import SessionState
from .ui import DemoView, fmt

log = logging.getLogger(__name__)

STATE_TEXT = {
    "created": "створено", "reference": "фіксація еталона", "generating": "генерування викликів",
    "challenge": "виконання виклику", "verifying": "перевірка виконання",
    "features": "обчислення ознак", "classifying": "класифікація",
    "bona_fide": "справжнє подання", "attack": "атака", "failed": "не пройдено",
}


@dataclass
class RunStats:
    """Накопичує час обробки кадрів для звіту про швидкодію."""

    frames: int = 0
    session_frames: int = 0
    total_ms: list = field(default_factory=list)       # уся обробка кадру (без відображення)
    pad_ms: list = field(default_factory=list)         # MediaPipe + SessionManager під час сеансу
    stage_ms: dict = field(default_factory=dict)
    decisions: list = field(default_factory=list)
    t_wall0: float = field(default_factory=time.perf_counter)

    def add(self, timings: dict, session_ms: float | None) -> None:
        """Додає час кадру; session_ms – час SessionManager.on_frame (None поза сеансом)."""
        self.frames += 1
        self.total_ms.append(timings["total_ms"] + (session_ms or 0.0))
        for k, v in timings.items():
            self.stage_ms.setdefault(k, []).append(v)
        if session_ms is not None:
            self.session_frames += 1
            self.pad_ms.append(timings["landmarks_ms"] + session_ms)

    def fps(self, last: int = 30) -> float:
        tail = self.total_ms[-last:]
        return 1000.0 / np.mean(tail) if tail else 0.0

    def summary(self) -> dict:
        wall = time.perf_counter() - self.t_wall0

        def stat(v):
            v = np.asarray(v, dtype=float)
            return {"mean": float(v.mean()), "p95": float(np.percentile(v, 95)),
                    "max": float(v.max())} if len(v) else None

        return {
            "frames": self.frames,
            "wall_s": wall,
            "fps_wall": self.frames / wall if wall > 0 else None,
            "fps_processing": 1000.0 / float(np.mean(self.total_ms)) if self.total_ms else None,
            "frame_ms": stat(self.total_ms),
            "pad_overhead_ms": stat(self.pad_ms),
            "stages_ms": {k: stat(v) for k, v in self.stage_ms.items()},
            "decisions": self.decisions,
        }


class DemoApp:
    """Обробляє відеопотік і проводить сеанси верифікації."""

    def __init__(self, cap: cv2.VideoCapture, pipeline: FramePipeline, manager: SessionManager,
                 enrolled: np.ndarray | None = None, person_id: str = "demo",
                 headless: bool = False, max_frames: int | None = None,
                 auto_start: bool = False, file_fps: float | None = None,
                 record_dir: Path | None = None, metadata: dict | None = None,
                 enroll_path: Path | None = None, stats_path: Path | None = None,
                 dataset_key: bytes | None = None):
        if record_dir is not None and dataset_key is None:
            raise ValueError("для запису набору потрібен ключ шифрування (FR-65)")
        if enroll_path is not None and dataset_key is None:
            raise ValueError("для збереження ембединга потрібен ключ шифрування (FR-65)")
        self.cap = cap
        self.pipeline = pipeline
        self.manager = manager
        self.enrolled = enrolled
        self.person_id = person_id
        self.headless = headless
        self.max_frames = max_frames
        self.auto_start = auto_start
        self.file_fps = file_fps
        self.record_dir = record_dir
        self.metadata = metadata or {}
        self.enroll_path = enroll_path
        self.stats_path = stats_path
        self.dataset_key = dataset_key
        self.view = DemoView()
        self.stats = RunStats()
        self.session = None
        self.writer = None
        self.rec_path: Path | None = None
        self.enroll_frames: list | None = None
        self.last_decision: dict | None = None
        self._final_frame_ms: float | None = None

    # ----- керування -----

    def start_session(self) -> None:
        """Починає новий сеанс (клавіша S)."""
        self.session = self.manager.start_session(self.person_id, "demo-operator", self.enrolled)
        self.last_decision = None
        if self.record_dir is not None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{self.metadata.get('participant', 'P00')}_{self.metadata.get('presentation_type', 'x')}_{stamp}"
            self.rec_path = self.record_dir / name
            self.rec_path.mkdir(parents=True, exist_ok=True)
        log.info("сеанс %d розпочато", self.session.id)

    def _open_writer(self, frame: np.ndarray, fps: float) -> None:
        h, w = frame.shape[:2]
        self.writer = cv2.VideoWriter(str(self.rec_path / "video.mp4"),
                                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def _finish_session(self) -> None:
        s = self.session
        dec = s.decision.to_dict()
        t_last_end = max((c["t_end"] for a in dec["explanation"]["attempts"] for c in a
                          if c.get("t_end") is not None), default=None)
        latency = {
            "decision_compute_ms": s.decision_latency_ms,       # класифікація
            "final_frame_ms": self._final_frame_ms,             # ознаки + класифікація на останньому кадрі
            "after_last_challenge_s": (s.last_t - t_last_end) if t_last_end is not None else None,
        }
        dec["explanation"]["latency"] = latency
        self.last_decision = dec
        self.stats.decisions.append({"session_id": s.id, "decision": dec["decision"],
                                     "score": dec["score"], **latency,
                                     "duration_s": dec["explanation"]["duration_s"]})
        log.info("сеанс %d: %s, оцінка %s, затримка %s", s.id, dec["decision"],
                 fmt(dec["score"]), latency)
        if self.rec_path is not None:
            self._encrypt_video()
            record = {"metadata": self.metadata, "settings": self.manager.settings.to_dict(),
                      "embedder": self.pipeline.embedder.name, "decision": dec}
            data = json.dumps(record, ensure_ascii=False, indent=1, default=_json_default).encode("utf-8")
            write_encrypted(self.rec_path / "session.json", data, self.dataset_key)
            log.info("запис збережено (AES-256-GCM): %s", self.rec_path)
            self.rec_path = None

    def _encrypt_video(self) -> None:
        """Закриває video.mp4, шифрує його у video.mp4.enc і видаляє відкритий файл."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        video = self.rec_path / "video.mp4"
        if video.exists():
            encrypt_file(video, self.dataset_key)

    def _close_recording(self) -> None:
        """Під час виходу: відео незавершеного сеансу теж не лишається відкритим."""
        if self.rec_path is not None:
            self._encrypt_video()
            log.warning("сеанс не завершено, зашифровано лише відео: %s", self.rec_path)
            self.rec_path = None
        elif self.writer is not None:
            self.writer.release()
            self.writer = None

    # ----- основний цикл -----

    def run(self) -> dict:
        """Обробляє кадри до натискання Q, кінця файлу або max_frames."""
        self.pipeline.warmup()
        self.stats = RunStats()
        t0 = time.perf_counter()
        idx = 0
        if self.auto_start:
            self.start_session()
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                t = idx / self.file_fps if self.file_fps else time.perf_counter() - t0
                idx += 1
                active = self.session is not None and not self.session.state.is_final
                data, timings = self.pipeline.process(
                    frame, t, landmarks=active or self.enroll_frames is not None,
                    hands=active and self._needs_hands())
                session_ms = None
                if active:
                    if self.rec_path is not None:
                        if self.writer is None:
                            self._open_writer(frame, self.file_fps or 30.0)
                        self.writer.write(frame)
                    tp = time.perf_counter()
                    self.manager.on_frame(self.session.id, data)
                    session_ms = (time.perf_counter() - tp) * 1000
                    if self.session.state.is_final:
                        self._final_frame_ms = session_ms
                        self._finish_session()
                self.stats.add(timings, session_ms)
                pad_ms = None if session_ms is None else timings["landmarks_ms"] + session_ms
                self._handle_enrollment(data)

                if not self.headless:
                    self.view.draw_landmarks(frame, data, self.pipeline.last_box)
                    cv2.imshow("FRS PAD demo", self.view.render(frame, self._panel_info(timings, pad_ms)))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("s") and not active:
                        self.start_session()
                    if key == ord("e") and not active:
                        self.enroll_frames = []
                if self.max_frames and idx >= self.max_frames:
                    break
                if self.auto_start and self.session is not None and self.session.state.is_final and self.headless:
                    break
        finally:
            self._close_recording()
        if not self.headless:
            cv2.destroyAllWindows()
        summary = self.stats.summary()
        if self.stats_path is not None:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            self.stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                  default=_json_default), encoding="utf-8")
        return summary

    def _needs_hands(self) -> bool:
        """Hands потрібен лише під час викликів з кистю (перекриття, проведення рукою)."""
        s = self.session
        if s.state not in (SessionState.GENERATING, SessionState.CHALLENGE):
            return False
        ch = s.current if s.state == SessionState.CHALLENGE else None
        return ch is None or ch.type != "turn_head"

    def _handle_enrollment(self, data) -> None:
        """Реєстрація обличчя заявленої особи протягом 30 кадрів (клавіша E).

        Ембединг зберігається у enroll_path.enc (AES-256-GCM); без enroll_path він
        лишається лише в пам'яті до виходу.
        """
        if self.enroll_frames is None:
            return
        self.enroll_frames.append(data)
        if len(self.enroll_frames) < 30:
            return
        emb = enroll_embedding(self.enroll_frames, self.manager.settings.min_det_score)
        self.enroll_frames = None
        if emb is None:
            log.warning("реєстрація не вдалася: обличчя не виявлено з потрібною упевненістю")
            return
        self.enrolled = emb
        if self.enroll_path is not None:
            self.enroll_path.parent.mkdir(parents=True, exist_ok=True)
            out = save_embedding(self.enroll_path, emb, self.dataset_key)
            log.info("ембединг заявленої особи збережено (AES-256-GCM): %s", out)

    def _panel_info(self, timings: dict, pad_ms: float | None) -> dict:
        info = {"metrics": [], "challenges": []}
        st = self.stats
        info["metrics"].append(f"FPS: {fmt(st.fps(), 1)}   кадр: {fmt(st.total_ms[-1] if st.total_ms else None, 1)} мс")
        info["metrics"].append(
            f"детекція {fmt(timings['detect_ms'], 1)} мс, ембединг {fmt(timings['embed_ms'], 1)} мс")
        if pad_ms is not None:
            info["metrics"].append(f"модуль виявлення атак: {fmt(pad_ms, 1)} мс/кадр")
        if self.last_decision is not None:
            lat = self.last_decision["explanation"].get("latency", {})
            info["metrics"].append(f"затримка рішення: {fmt(lat.get('decision_compute_ms'), 2)} мс "
                                   f"(після виклику {fmt(lat.get('after_last_challenge_s'), 2)} с)")
        if self.enrolled is None:
            info["metrics"].append("обличчя не зареєстровано (E): s_enr = s_ref")
        if self.enroll_frames is not None:
            info["status"] = {"instruction": "Реєстрація обличчя: дивіться в камеру",
                              "progress": len(self.enroll_frames) / 30}
        s = self.session
        if s is None:
            return info
        status = self.manager.status(s)
        info.setdefault("status", status)
        info["state_text"] = STATE_TEXT.get(status["state"], status["state"])
        t, s_ref, s_enr = s.series_arrays()
        t0 = s.t_started or 0.0
        info["series"] = {"t": t - t0, "s_ref": s_ref, "s_enr": s_enr}
        info["spans"] = [(c.t_start - t0, (c.t_end or s.last_t) - t0) for c in s.challenges
                         if c.t_start is not None]
        for i, c in enumerate(s.challenges):
            mark = "+" if c.performed_at is not None else ("×" if c.t_end is not None else "…")
            score = f"  p = {fmt(c.score)}" if c.score is not None else ""
            info["challenges"].append(f"{i + 1}. {c.instruction()} {mark}{score}")
        if self.last_decision is not None:
            info["decision"] = self.last_decision
        return info


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, SessionState):
        return obj.value
    return str(obj)
