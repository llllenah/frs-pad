"""Параметри модуля виявлення атак (клас PadSettings на рисунку 3.10).

Значення за замовчуванням узято з технічного завдання (підпункт 4.2.3) та
підрозділу 3.3 пояснювальної записки. Параметри можна задати у YAML-файлі
(configs/pad.yaml), адміністратор змінює їх у межах, перевірених validate().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

#: Типи викликів (поле challenge_type таблиці session_challenges).
CHALLENGE_TYPES = ("cover_left", "cover_right", "cover_mouth", "turn_head", "hand_swipe")

#: Найменша кількість типів у наборі (ТЗ, п. 4.2.3: не менше 4 типів).
MIN_CHALLENGE_TYPES = 4


@dataclass
class PadSettings:
    """Налаштування сеансу верифікації з активними викликами."""

    # Параметри, які задає адміністратор (FR-62).
    challenge_types: list[str] = field(default_factory=lambda: list(CHALLENGE_TYPES))
    challenge_count: int = 3
    duration_s: float = 5.0
    threshold: float = 0.5
    max_retries: int = 2

    # Параметри алгоритму (підрозділ 3.3).
    reference_s: float = 1.0          # тривалість фіксації еталона
    reference_timeout_s: float = 3.0  # скільки чекати кадрів з упевненістю >= min_det_score
    min_det_score: float = 0.9        # упевненість детекції для кадрів еталона, формула (3.1)
    before_s: float = 1.0             # вікно «до»
    after_s: float = 1.5              # вікно «після»
    hold_s: float = 0.5               # скільки тримати виконаний виклик перед завершенням
    session_timeout_s: float = 20.0   # найбільша тривалість однієї спроби сеансу
    min_coverage: float = 0.4         # поріг частки перекриття, формула (3.3)
    min_yaw_deg: float = 30.0         # поріг кута повороту голови
    min_shift: float = 0.5            # зміщення кисті при проведенні, частка ширини обличчя
    consecutive_frames: int = 5       # кадрів поспіль для перекриття і повороту
    recovery_ratio: float = 0.9       # рівень відновлення, формула (3.4)
    face_hold_s: float = 1.0          # скільки використовувати останні точки обличчя, якщо його не знайдено

    def validate(self) -> "PadSettings":
        """Перевіряє межі параметрів (FR-62) і повертає сам об'єкт."""
        unknown = set(self.challenge_types) - set(CHALLENGE_TYPES)
        if unknown:
            raise ValueError(f"невідомі типи викликів: {sorted(unknown)}")
        if len(set(self.challenge_types)) < MIN_CHALLENGE_TYPES:
            raise ValueError(f"потрібно не менше {MIN_CHALLENGE_TYPES} типів викликів")
        if not 2 <= self.challenge_count <= 5:
            raise ValueError("кількість викликів має бути в межах 2–5")
        if not 3.0 <= self.duration_s <= 5.0:
            raise ValueError("тривалість виклику має бути в межах 3–5 с")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("поріг рішення має бути в межах [0; 1]")
        if not 0 <= self.max_retries <= 2:
            raise ValueError("кількість повторів – не більше 2")
        if self.session_timeout_s > 20.0:
            raise ValueError("тривалість сеансу – не більше 20 с")
        return self

    def to_dict(self) -> dict:
        """Повертає параметри у вигляді словника (для журналу сеансу)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PadSettings":
        """Створює параметри зі словника, ігноруючи невідомі ключі."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names}).validate()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PadSettings":
        """Читає параметри з YAML-файлу."""
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data.get("pad", data))
