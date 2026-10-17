"""Керування сеансом верифікації з активними викликами (SessionManager).

Переходи між станами відповідають діаграмі станів (рисунок 3.11):
Створено -> [Активний сеанс: Фіксація еталона -> Генерування викликів ->
Виконання виклику -> Перевірка виконання -> Обчислення ознак -> ...] ->
Класифікація -> Справжнє подання | Атака, або Не пройдено.

Час відлічується за мітками кадрів FrameData.t, тому той самий код працює і
з вебкамерою, і з записаним відео, і з синтетичними кадрами в тестах.
"""

from __future__ import annotations

import itertools
import logging
import time

import numpy as np

from .challenges import Challenge, ChallengeGenerator
from .classifiers import AttackClassifier, session_score
from .features import DriftFeatureExtractor, InsufficientDataError, l2_normalize, reference_embedding
from .settings import PadSettings
from .structures import FrameData, SessionDecision, SessionState
from .verifier import ChallengeVerifier

log = logging.getLogger(__name__)

#: Найменша кількість кадрів з упевненістю >= min_det_score для еталона.
MIN_REFERENCE_FRAMES = 3

FAILURE_TEXT = {
    "timeout": "перевищено тривалість сеансу",
    "no_face": "обличчя не виявлено",
    "not_performed": "виклики не виконано",
    "error": "збій модуля виявлення атак",
}


class VerificationSession:
    """Стан одного сеансу верифікації."""

    def __init__(self, session_id: int, claimed_person_id, operator_id,
                 enrolled: np.ndarray | None):
        self.id = session_id
        self.claimed_person_id = claimed_person_id
        self.operator_id = operator_id
        self.enrolled = l2_normalize(enrolled) if enrolled is not None else None
        self.attempt_no = 1
        self.state = SessionState.CREATED
        self.reference: np.ndarray | None = None
        self.challenges: list[Challenge] = []
        self.index = 0
        self.attempts: list[list[dict]] = []
        self.t_started: float | None = None
        self.t_attempt: float | None = None
        self.t_state: float | None = None
        self.last_t: float | None = None
        self.failure_reason: str | None = None
        self.decision: SessionDecision | None = None
        self.decision_latency_ms: float | None = None
        self.transitions: list[tuple[float | None, SessionState]] = [(None, self.state)]
        self._ref_buffer: list[tuple[float, np.ndarray, float]] = []
        self._pending: list[np.ndarray | None] = []  # ембединги до фіксації еталона
        self.t: list[float] = []
        self.s_ref: list[float] = []
        self.s_enr: list[float] = []

    @property
    def current(self) -> Challenge | None:
        """Поточний виклик або None."""
        return self.challenges[self.index] if self.index < len(self.challenges) else None

    def series_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ряди t, s_ref(t), s_enr(t) від початку сеансу."""
        return np.array(self.t), np.array(self.s_ref), np.array(self.s_enr)


class SessionManager:
    """Точка входу модуля: створює сеанси, отримує кадри, формує рішення."""

    def __init__(self, classifier: AttackClassifier, settings: PadSettings | None = None,
                 verifier: ChallengeVerifier | None = None,
                 features: DriftFeatureExtractor | None = None,
                 generator: ChallengeGenerator | None = None):
        self.settings = (settings or PadSettings()).validate()
        self.classifier = classifier
        self.verifier = verifier or ChallengeVerifier()
        self.features = features or DriftFeatureExtractor(
            self.settings.before_s, self.settings.after_s, self.settings.recovery_ratio)
        self.generator = generator or ChallengeGenerator(self.settings)
        self.sessions: dict[int, VerificationSession] = {}
        self._ids = itertools.count(1)

    # ----- публічний інтерфейс (рисунок 3.10) -----

    def start_session(self, person_id, operator_id=None,
                      enrolled_embedding: np.ndarray | None = None) -> VerificationSession:
        """Створює сеанс для заявленої особи. Перший кадр запускає фіксацію еталона."""
        s = VerificationSession(next(self._ids), person_id, operator_id, enrolled_embedding)
        self.sessions[s.id] = s
        return s

    def on_frame(self, session_id: int, frame: FrameData) -> None:
        """Обробляє черговий кадр сеансу."""
        s = self.sessions[session_id]
        if s.state.is_final:
            return
        try:
            self._step(s, frame)
        except Exception as exc:  # збій модуля: успішне рішення за замовчуванням не формується
            log.exception("збій модуля виявлення атак")
            self._fail(s, frame.t, "error", detail=repr(exc))

    def decide(self, session: VerificationSession) -> SessionDecision:
        """Класифікує виконані виклики та формує рішення сеансу."""
        t0 = time.perf_counter()
        X = np.stack([ch.features.values for ch in session.challenges])
        scores = self.classifier.predict_batch(X)
        if not np.all(np.isfinite(scores)):
            raise ValueError("класифікатор повернув некоректну оцінку")
        for ch, p in zip(session.challenges, scores):
            ch.score = float(p)
        score = session_score(scores)
        is_attack = score >= self.settings.threshold
        session.decision_latency_ms = (time.perf_counter() - t0) * 1000.0
        self._enter(session, SessionState.ATTACK if is_attack else SessionState.BONA_FIDE,
                    session.last_t)
        session.attempts.append([ch.summary() for ch in session.challenges])
        session.decision = SessionDecision(
            session_id=session.id, decision=session.state.value, score=score,
            is_attack=bool(is_attack), threshold=self.settings.threshold,
            explanation=self._explanation(session))
        return session.decision

    def status(self, session: VerificationSession, t: float | None = None) -> dict:
        """Поточна інструкція, залишок часу і прогрес для екранної форми."""
        st = session.state
        t = session.last_t if t is None else t
        info = {"state": st.value, "attempt": session.attempt_no,
                "max_attempts": self.settings.max_retries + 1, "remaining": None,
                "progress": 0.0, "index": session.index, "count": len(session.challenges)}
        if st in (SessionState.CREATED, SessionState.REFERENCE):
            info["instruction"] = "Дивіться прямо в камеру"
        elif st == SessionState.GENERATING:
            info["instruction"] = (f"Новий набір викликів (спроба {session.attempt_no})"
                                   if session.attempt_no > 1 else "Приготуйтеся")
        elif st == SessionState.CHALLENGE and session.current is not None:
            ch = session.current
            info["instruction"] = ch.instruction()
            if ch.t_start is not None and t is not None:
                info["remaining"] = max(0.0, ch.duration_s - (t - ch.t_start))
            info["progress"] = ch.progress()
        elif st == SessionState.FEATURES:
            info["instruction"] = "Опустіть руку і дивіться прямо в камеру"
            info["progress"] = 1.0
        elif st == SessionState.FAILED:
            info["instruction"] = "Не пройдено: " + FAILURE_TEXT.get(session.failure_reason, "")
        elif st == SessionState.ATTACK:
            info["instruction"] = "Атака підміни обличчя"
        elif st == SessionState.BONA_FIDE:
            info["instruction"] = "Справжнє подання"
        else:
            info["instruction"] = ""
        return info

    # ----- внутрішня логіка -----

    def _enter(self, s: VerificationSession, state: SessionState, t: float | None) -> None:
        s.state = state
        s.t_state = t
        s.transitions.append((t, state))

    def _record_similarity(self, s: VerificationSession, frame: FrameData) -> None:
        s.t.append(frame.t)
        if s.reference is None:
            s._pending.append(frame.embedding)
            s.s_ref.append(float("nan"))
            s.s_enr.append(float("nan"))
            return
        s.s_ref.append(self.features.similarity(frame.embedding, s.reference))
        s.s_enr.append(self.features.similarity(frame.embedding, s.enrolled))

    def _fix_reference(self, s: VerificationSession, t: float) -> bool:
        window = [(e, sc) for (tt, e, sc) in s._ref_buffer if tt >= t - self.settings.reference_s]
        good = sum(1 for _, sc in window if sc >= self.settings.min_det_score)
        ref = reference_embedding([e for e, _ in window], [sc for _, sc in window],
                                  self.settings.min_det_score)
        if ref is None or good < MIN_REFERENCE_FRAMES:
            return False
        s.reference = ref
        if s.enrolled is None:  # без ембединга з бази s_enr збігається з s_ref
            log.warning("ембединг заявленої особи не задано, s_enr = s_ref")
            s.enrolled = ref
        for i, e in enumerate(s._pending):
            s.s_ref[i] = self.features.similarity(e, s.reference)
            s.s_enr[i] = self.features.similarity(e, s.enrolled)
        s._pending.clear()
        s._ref_buffer.clear()
        return True

    def _step(self, s: VerificationSession, frame: FrameData) -> None:
        t = frame.t
        st = self.settings
        if s.state == SessionState.CREATED:
            s.t_started = s.t_attempt = t
            self._enter(s, SessionState.REFERENCE, t)
        s.last_t = t
        if t - s.t_attempt > st.session_timeout_s:
            self._fail(s, t, "timeout")
            return
        self._record_similarity(s, frame)

        if s.state == SessionState.REFERENCE:
            if frame.embedding is not None:
                s._ref_buffer.append((t, frame.embedding, frame.det_score))
            elapsed = t - s.t_state
            if elapsed >= st.reference_s and self._fix_reference(s, t):
                self._enter(s, SessionState.GENERATING, t)
            elif elapsed >= st.reference_timeout_s:
                self._fail(s, t, "no_face")
                return

        if s.state == SessionState.GENERATING:
            if not s.challenges:
                prev = s.attempts[-1][-1]["challenge_type"] if s.attempts else None
                s.challenges = self.generator.generate(st.challenge_count, st.challenge_types, prev)
                s.index = 0
            wait = st.before_s if s.attempt_no > 1 else 0.0  # чисте вікно «до» після невдалої спроби
            if t - s.t_state >= wait:
                self._enter(s, SessionState.CHALLENGE, t)
            return

        if s.state == SessionState.CHALLENGE:
            ch = s.current
            ch.add_frame(frame, self.verifier)
            held = ch.performed_at is not None and t - ch.performed_at >= st.hold_s
            if held or t - ch.t_start >= ch.duration_s:
                ch.t_end = t
                self._enter(s, SessionState.VERIFYING, t)
                if ch.is_performed(self.verifier):
                    self._enter(s, SessionState.FEATURES, t)
                else:
                    self._not_performed(s, t)
            return

        if s.state == SessionState.FEATURES:
            ch = s.current
            if t - ch.t_end < st.after_s:
                return
            try:
                ch.features = self.features.extract(s, ch)
            except InsufficientDataError as exc:
                log.info("ознаки не обчислено: %s", exc)
                self._not_performed(s, t)
                return
            s.index += 1
            if s.index < len(s.challenges):
                self._enter(s, SessionState.CHALLENGE, t)
            else:
                self._enter(s, SessionState.CLASSIFYING, t)
                self.decide(s)

    def _not_performed(self, s: VerificationSession, t: float) -> None:
        s.attempts.append([ch.summary() for ch in s.challenges[: s.index + 1]])
        if s.attempt_no - 1 < self.settings.max_retries:
            s.attempt_no += 1
            s.t_attempt = t
            s.challenges = []
            s.index = 0
            self._enter(s, SessionState.GENERATING, t)
        else:
            self._fail(s, t, "not_performed")

    def _fail(self, s: VerificationSession, t: float, reason: str, detail: str | None = None) -> None:
        if s.challenges and reason != "not_performed":
            s.attempts.append([ch.summary() for ch in s.challenges[: s.index + 1]])
        s.failure_reason = reason
        self._enter(s, SessionState.FAILED, t)
        expl = self._explanation(s)
        expl["reason"] = reason
        if detail:
            expl["detail"] = detail
        s.decision = SessionDecision(session_id=s.id, decision="failed", score=None,
                                     is_attack=None, threshold=self.settings.threshold,
                                     explanation=expl)

    def _explanation(self, s: VerificationSession) -> dict:
        t, s_ref, s_enr = s.series_arrays()
        t0 = s.t_started or 0.0
        return {
            "claimed_person_id": s.claimed_person_id,
            "operator_id": s.operator_id,
            "attempt_no": s.attempt_no,
            "model_version": getattr(self.classifier, "version", "unknown"),
            "attempts": s.attempts,
            "decision_latency_ms": s.decision_latency_ms,
            "duration_s": (s.last_t - t0) if s.last_t is not None else None,
            "t0": t0,  # часові мітки викликів у attempts – абсолютні, ряд series – від t0
            "series": {"t": _clean(t - t0), "s_ref": _clean(s_ref), "s_enr": _clean(s_enr)},
        }


def _clean(values: np.ndarray) -> list:
    """Округлює ряд і замінює NaN на None, щоб отримати коректний JSON."""
    return [round(float(v), 4) if np.isfinite(v) else None for v in values]
