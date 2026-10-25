"""Ідентифікація 1:N для вкладки «Моніторинг»: пошук обличчя серед усіх еталонів.

Охоронець бачить потік камери: кожне обличчя в кадрі порівнюється з еталонами
всіх зареєстрованих осіб (data/enrolled/*.npy.enc і еталони в пам'яті сервера).
Подібність – косинусна, як s_enr у модулі виявлення атак: ембединги
L2-нормовані, тому вона дорівнює скалярному добутку. Якщо найбільша подібність
не менша за поріг, обличчя вважається відомим (зелена рамка з псевдонімом),
інакше – невідомим (червона рамка і сповіщення «Невідома особа»).

Пошук – повний перебір: FAISS IndexFlatIP, якщо пакет faiss встановлено,
інакше матричний добуток numpy (результат той самий). Індекс FAISS IVF з ТЗ
потрібен для бази з тисяч осіб у повній системі (дипломний проєкт); для
кількох учасників прототипу точний перебір швидший і не потребує навчання індексу.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, fields

import numpy as np

from .features import l2_normalize

log = logging.getLogger(__name__)

try:  # необов'язкова залежність
    import faiss
except ImportError:  # pragma: no cover - залежить від середовища
    faiss = None


@dataclass
class MonitorSettings:
    """Параметри моніторингу (ключ monitoring у configs/pad.yaml).

    Поріг 0,5 обрано з запасом над порогом верифікації MobileFaceNet INT8 на LFW
    (≈ 0,30 за найкращої точності, runs/lfw_mobilefacenet_int8.json). Для 1:N поріг
    має бути вищим, ніж для 1:1: обличчя порівнюється з кожною особою бази, і
    ймовірність випадкового збігу зростає з кількістю осіб. Еталон і кадр з тієї
    самої камери зазвичай дають набагато більшу подібність, ніж пари LFW з різних
    фото (на скріншотах форми 4.7 s_enr ≈ 0,98), тому своїх поріг 0,5 майже не
    відкидає. На записах учасників поріг ще треба перевірити.
    """

    threshold: float = 0.5        # найменша косинусна подібність для «відомої особи»
    min_det_score: float = 0.9    # обличчя з меншою упевненістю детекції не розпізнаються
    max_faces: int = 5            # скільки найбільших облич кадру обробляти
    confirm_frames: int = 3       # кадрів поспіль, щоб подія потрапила в журнал
    event_gap_s: float = 10.0     # нова подія для особи – якщо її не було в кадрі довше, с
    max_events: int = 100         # скільки подій тримати в журналі

    def validate(self) -> "MonitorSettings":
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("поріг розпізнавання має бути в межах [0; 1]")
        if not 0.0 <= self.min_det_score <= 1.0:
            raise ValueError("упевненість детекції має бути в межах [0; 1]")
        if self.max_faces < 1 or self.confirm_frames < 1 or self.max_events < 1:
            raise ValueError("max_faces, confirm_frames і max_events мають бути додатними")
        if self.event_gap_s < 0:
            raise ValueError("event_gap_s не може бути від'ємним")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "MonitorSettings":
        """Створює параметри зі словника, ігноруючи невідомі ключі."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in names}).validate()


@dataclass
class Match:
    """Результат пошуку для одного обличчя."""

    person_id: str | None     # псевдонім, якщо подібність >= порогу, інакше None («невідома»)
    similarity: float         # найбільша косинусна подібність (NaN, якщо база порожня)
    nearest: str | None       # найближчий еталон незалежно від порогу

    @property
    def known(self) -> bool:
        return self.person_id is not None


class FaceGallery:
    """Еталони зареєстрованих осіб і пошук найближчого (top-1) для ембедингів кадру."""

    def __init__(self, threshold: float = 0.5, use_faiss: bool | None = None):
        self.threshold = threshold
        self.use_faiss = (faiss is not None) if use_faiss is None else (use_faiss and faiss is not None)
        self._items: dict[str, np.ndarray] = {}
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None
        self._index = None

    @property
    def backend(self) -> str:
        return "FAISS IndexFlatIP" if self.use_faiss else "numpy (точний перебір)"

    @property
    def ids(self) -> list[str]:
        return sorted(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, person_id: str) -> bool:
        return person_id in self._items

    def replace(self, items: dict[str, np.ndarray]) -> None:
        """Замінює всю базу еталонів (після перечитування data/enrolled)."""
        self._items = {}
        for pid, emb in items.items():
            self._put(pid, emb)
        self._build()

    def add(self, person_id: str, emb: np.ndarray) -> None:
        """Додає або оновлює еталон однієї особи (нова реєстрація)."""
        self._put(person_id, emb)
        self._build()

    def _put(self, person_id: str, emb: np.ndarray) -> None:
        v = np.asarray(emb, dtype=np.float32).reshape(-1)
        dim = next(iter(self._items.values())).shape[0] if self._items else v.shape[0]
        if v.shape[0] != dim or not np.all(np.isfinite(v)):
            log.warning("еталон %s пропущено: розмірність %d (очікується %d) або NaN", person_id,
                        v.shape[0], dim)
            return
        self._items[person_id] = l2_normalize(v).astype(np.float32)

    def _build(self) -> None:
        self._ids = sorted(self._items)
        if not self._ids:
            self._matrix = self._index = None
            return
        self._matrix = np.ascontiguousarray(np.stack([self._items[i] for i in self._ids]))
        if self.use_faiss:
            self._index = faiss.IndexFlatIP(self._matrix.shape[1])
            self._index.add(self._matrix)

    def search(self, embeddings: np.ndarray) -> list[Match]:
        """Найближчий еталон для кожного рядка embeddings (k × d)."""
        q = np.asarray(embeddings, dtype=np.float32)
        if q.ndim == 1:
            q = q[None]
        if len(q) == 0:
            return []
        if self._matrix is None or q.shape[1] != self._matrix.shape[1]:
            if self._matrix is not None:
                log.warning("розмірність ембедингів кадру %d не збігається з базою (%d): інша модель?",
                            q.shape[1], self._matrix.shape[1])
            return [Match(None, float("nan"), None) for _ in range(len(q))]
        q = np.ascontiguousarray(q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12))
        if self._index is not None:
            sims, idx = self._index.search(q, 1)
            best, best_i = sims[:, 0], idx[:, 0]
        else:
            s = q @ self._matrix.T
            best_i = s.argmax(axis=1)
            best = s[np.arange(len(q)), best_i]
        out = []
        for sim, i in zip(best, best_i):
            pid = self._ids[int(i)]
            out.append(Match(pid if sim >= self.threshold else None, float(sim), pid))
        return out
