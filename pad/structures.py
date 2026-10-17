"""Структури даних модуля: кадр, стан сеансу, вектор ознак, рішення."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np


@dataclass
class FrameData:
    """Дані одного кадру, які конвеєр передає модулю виявлення атак.

    Координати ключових точок задано в пікселях кадру, що надходить з камери
    без дзеркального відображення.
    """

    t: float                                 # час кадру, с
    embedding: np.ndarray | None = None      # L2-нормований ембединг (128 значень)
    det_score: float = 0.0                   # упевненість детекції обличчя
    face: np.ndarray | None = None           # (468, 2) точки Face Mesh
    hand: np.ndarray | None = None           # (21, 2) точки Hands
    size: tuple[int, int] = (1280, 720)      # (ширина, висота) кадру


class SessionState(str, Enum):
    """Стани сеансу верифікації (рисунок 3.11)."""

    CREATED = "created"          # Створено
    REFERENCE = "reference"      # Фіксація еталона
    GENERATING = "generating"    # Генерування викликів
    CHALLENGE = "challenge"      # Виконання виклику
    VERIFYING = "verifying"      # Перевірка виконання
    FEATURES = "features"        # Обчислення ознак (разом зі збором вікна «після»)
    CLASSIFYING = "classifying"  # Класифікація
    BONA_FIDE = "bona_fide"      # Справжнє подання
    ATTACK = "attack"            # Атака
    FAILED = "failed"            # Не пройдено

    @property
    def is_final(self) -> bool:
        """Чи є стан кінцевим."""
        return self in (SessionState.BONA_FIDE, SessionState.ATTACK, SessionState.FAILED)


@dataclass
class FeatureVector:
    """Вектор ознак одного виклику (26 значень)."""

    values: np.ndarray
    names: list[str]

    def as_dict(self) -> dict[str, float]:
        """Повертає ознаки у вигляді словника «назва – значення»."""
        return {n: float(v) for n, v in zip(self.names, self.values)}


@dataclass
class SessionDecision:
    """Рішення модуля виявлення атак для сеансу."""

    session_id: int
    decision: str                     # bona_fide, attack або failed
    score: float | None = None        # оцінка атаки в [0; 1]
    is_attack: bool | None = None
    threshold: float | None = None
    explanation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Повертає рішення у вигляді словника для запису в JSON."""
        return asdict(self)

    def to_json(self) -> str:
        """Серіалізує рішення у формат JSON (вихідні дані за ТЗ)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"не вдається серіалізувати {type(obj)}")
