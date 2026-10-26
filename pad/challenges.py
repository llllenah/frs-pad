"""Активні виклики та їх генерування (Challenge, ChallengeGenerator).

Кожен виклик накопичує кадри, на яких він виконувався, і для кожного кадру
зберігає виміряні значення (частку перекриття, кут yaw або зміщення кисті).
Метод is_performed() перевіряє правило виконання за цими значеннями.
"""

from __future__ import annotations

import secrets
from abc import ABC, abstractmethod

import numpy as np

from .settings import CHALLENGE_TYPES, PadSettings
from .structures import FeatureVector, FrameData
from .verifier import ChallengeVerifier


class Challenge(ABC):
    """Абстрактний виклик з типом, тривалістю та списком кадрів."""

    type: str = ""

    def __init__(self, duration_s: float = 5.0, consecutive_frames: int = 5,
                 face_hold_s: float = 1.0):
        self.duration_s = duration_s
        self.consecutive_frames = consecutive_frames
        self.face_hold_s = face_hold_s
        self.frames: list[FrameData] = []
        self.measures: list[dict] = []
        self.t_start: float | None = None
        self.t_end: float | None = None
        self.performed_at: float | None = None
        self.features: FeatureVector | None = None
        self.score: float | None = None
        self._last_face: tuple[float, np.ndarray] | None = None

    @abstractmethod
    def instruction(self) -> str:
        """Текст інструкції для користувача."""

    @abstractmethod
    def measure(self, face: np.ndarray | None, hand: np.ndarray | None,
                frame: FrameData, verifier: ChallengeVerifier) -> dict:
        """Вимірює значення для одного кадру; ключ «ok» – чи виконано умову."""

    def _face_for(self, frame: FrameData) -> np.ndarray | None:
        """Точки обличчя кадру або останні відомі, якщо обличчя закрито рукою."""
        if frame.face is not None:
            self._last_face = (frame.t, frame.face)
            return frame.face
        if self._last_face is not None and frame.t - self._last_face[0] <= self.face_hold_s:
            return self._last_face[1]
        return None

    def add_frame(self, frame: FrameData, verifier: ChallengeVerifier) -> dict:
        """Додає кадр до виклику та оновлює момент виконання."""
        if self.t_start is None:
            self.t_start = frame.t
        m = self.measure(self._face_for(frame), frame.hand, frame, verifier)
        m["t"] = frame.t
        self.frames.append(frame)
        self.measures.append(m)
        if self.performed_at is None and self.is_performed(verifier):
            self.performed_at = frame.t
        return m

    def _longest_run(self) -> int:
        best = run = 0
        for m in self.measures:
            if m.get("held") and m.get("ok"):
                continue  # кадр без точок, «утриманий» після виконання: серію не перериває, але й не подовжує
            run = run + 1 if m.get("ok") else 0
            best = max(best, run)
        return best

    def is_performed(self, verifier: ChallengeVerifier | None = None) -> bool:
        """Умову виконано протягом не менше consecutive_frames кадрів поспіль."""
        return self._longest_run() >= self.consecutive_frames

    def progress(self) -> float:
        """Частка виконання від 0 до 1 для індикатора в інтерфейсі."""
        return min(1.0, self._longest_run() / max(self.consecutive_frames, 1))

    def summary(self) -> dict:
        """Результат виклику для журналу (таблиця session_challenges)."""
        return {
            "challenge_type": self.type,
            "instruction": self.instruction(),
            "performed": self.performed_at is not None,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "performed_at": self.performed_at,
            "score": self.score,
            "features": self.features.as_dict() if self.features is not None else None,
        }


class OcclusionChallenge(Challenge):
    """Закрити долонею ліву чи праву половину обличчя або рот і підборіддя."""

    INSTRUCTIONS = {
        "left": "Закрийте долонею ліву половину обличчя",
        "right": "Закрийте долонею праву половину обличчя",
        "mouth": "Закрийте долонею рот і підборіддя",
    }

    def __init__(self, side: str, min_coverage: float = 0.4, **kw):
        super().__init__(**kw)
        if side not in self.INSTRUCTIONS:
            raise ValueError(f"невідома сторона перекриття: {side}")
        self.side = side
        self.min_coverage = min_coverage
        self.type = f"cover_{side}"

    def instruction(self) -> str:
        return self.INSTRUCTIONS[self.side]

    def measure(self, face, hand, frame, verifier) -> dict:
        info = verifier.coverage_info(face, hand)
        # частка рахується від площі тієї частини обличчя, яку треба закрити: долоня закриває
        # лише 0,2–0,4 площі всього обличчя, тож поріг від усього обличчя для людей недосяжний
        if self.side == "mouth":
            # рот і підборіддя: частка нижньої частини обличчя (нижче кінчика носа)
            value, side_ok = info["lower_part"], info["lower"]
        else:
            # половина обличчя: частка площі цієї половини
            value, side_ok = info[f"half_{self.side}"], info["side"] == self.side
        info["value"] = value
        info["ok"] = bool(value >= self.min_coverage and side_ok)
        return info

    def summary(self) -> dict:
        s = super().summary()
        s["coverage"] = max((m["value"] for m in self.measures), default=0.0)
        return s


class HeadTurnChallenge(Challenge):
    """Повернути голову ліворуч або праворуч на кут не менше min_yaw_deg."""

    type = "turn_head"

    def __init__(self, direction: str, min_yaw_deg: float = 30.0, **kw):
        super().__init__(**kw)
        if direction not in ("left", "right"):
            raise ValueError(f"невідомий напрям повороту: {direction}")
        self.direction = direction
        self.min_yaw_deg = min_yaw_deg

    def instruction(self) -> str:
        return "Поверніть голову " + ("ліворуч" if self.direction == "left" else "праворуч")

    def measure(self, face, hand, frame, verifier) -> dict:
        if frame.face is None:  # для повороту потрібні точки саме цього кадру
            # при повороті на 60° і більше Face Mesh втрачає обличчя в профіль: якщо перед цим
            # кут уже перевищив поріг, такі кадри не переривають серію (не довше face_hold_s), але й
            # не зараховуються – потрібні consecutive_frames кадрів з виміряним кутом
            last = next((m for m in reversed(self.measures) if m.get("yaw") is not None), None)
            held = bool(last is not None and last["ok"] and frame.t - last["t"] <= self.face_hold_s)
            return {"yaw": None, "ok": held, "held": True}
        yaw, pitch, roll = verifier.head_pose(frame.face, frame.size)
        sign = 1.0 if self.direction == "left" else -1.0
        return {"yaw": yaw, "pitch": pitch, "roll": roll,
                "ok": bool(np.isfinite(yaw) and sign * yaw >= self.min_yaw_deg)}

    def summary(self) -> dict:
        s = super().summary()
        s["direction"] = self.direction
        yaws = [abs(m["yaw"]) for m in self.measures if m.get("yaw") is not None]
        s["yaw_deg"] = max(yaws, default=None)
        return s


class HandSwipeChallenge(Challenge):
    """Провести рукою перед обличчям: центр кисті перетинає середню лінію обличчя."""

    type = "hand_swipe"

    def __init__(self, min_shift: float = 0.5, **kw):
        super().__init__(**kw)
        self.min_shift = min_shift

    def instruction(self) -> str:
        return "Проведіть рукою перед обличчям"

    def measure(self, face, hand, frame, verifier) -> dict:
        return {"offset": verifier.hand_offset(face, hand), "ok": False}

    def is_performed(self, verifier: ChallengeVerifier | None = None) -> bool:
        """Кисть побувала з одного боку середньої лінії, а потім з іншого.

        Сумарне зміщення між крайніми положеннями – не менше min_shift ширини обличчя.
        """
        half = self.min_shift / 2.0
        seen_left = seen_right = False
        for m in self.measures:
            d = m.get("offset")
            if d is None:
                continue
            if d >= half:
                if seen_right:
                    return True
                seen_left = True
            elif d <= -half:
                if seen_left:
                    return True
                seen_right = True
        return False

    def progress(self) -> float:
        offsets = [m["offset"] for m in self.measures if m.get("offset") is not None]
        if self.is_performed() or not offsets:
            return 1.0 if self.is_performed() else 0.0
        return min(0.99, (max(offsets) - min(offsets)) / max(self.min_shift, 1e-6))


def make_challenge(kind: str, settings: PadSettings, rng) -> Challenge:
    """Створює виклик заданого типу з параметрами settings."""
    common = {"duration_s": settings.duration_s,
              "consecutive_frames": settings.consecutive_frames,
              "face_hold_s": settings.face_hold_s}
    if kind.startswith("cover_"):
        return OcclusionChallenge(kind.removeprefix("cover_"), settings.min_coverage, **common)
    if kind == "turn_head":
        return HeadTurnChallenge(rng.choice(["left", "right"]), settings.min_yaw_deg, **common)
    if kind == "hand_swipe":
        return HandSwipeChallenge(settings.min_shift, **common)
    raise ValueError(f"невідомий тип виклику: {kind}")


class ChallengeGenerator:
    """Формує випадкову послідовність викликів.

    Використовує криптографічно стійкий генератор secrets.SystemRandom, щоб
    послідовність не можна було передбачити і записати відповідь заздалегідь.
    Два однакові типи поспіль не допускаються.
    """

    def __init__(self, settings: PadSettings | None = None, rng=None):
        self.settings = settings or PadSettings()
        self.rng = rng if rng is not None else secrets.SystemRandom()

    def generate_types(self, n: int | None = None, types: list[str] | None = None,
                       previous: str | None = None) -> list[str]:
        """Повертає n типів викликів без двох однакових поспіль.

        previous – тип останнього виклику попередньої спроби, з якого не можна починати.
        """
        n = self.settings.challenge_count if n is None else n
        types = list(dict.fromkeys(types or self.settings.challenge_types))
        if n < 1:
            raise ValueError("кількість викликів має бути додатною")
        unknown = set(types) - set(CHALLENGE_TYPES)
        if unknown:
            raise ValueError(f"невідомі типи викликів: {sorted(unknown)}")
        if len(types) < 2 and (n > 1 or previous in types):
            raise ValueError("для послідовності без повторів потрібно щонайменше 2 типи")
        result: list[str] = []
        prev = previous
        for _ in range(n):
            kind = self.rng.choice([t for t in types if t != prev])
            result.append(kind)
            prev = kind
        return result

    def generate(self, n: int | None = None, types: list[str] | None = None,
                 previous: str | None = None) -> list[Challenge]:
        """Повертає список об'єктів-викликів (метод generate на рисунку 3.10)."""
        return [make_challenge(k, self.settings, self.rng)
                for k in self.generate_types(n, types, previous)]
