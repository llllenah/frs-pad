"""Детектори облич з 5 ключовими точками.

Основний детектор – MTCNN (pad/mtcnn.py, моделі mtcnn_*.onnx). YuNet з OpenCV Zoo
лишається запасним варіантом: як і MTCNN, він повертає рамку, упевненість і
5 точок (очі, кінчик носа, кутики рота), тому решта коду від вибору детектора
не залежить. Якщо моделей немає, точки беруться з Face Mesh (MeshDetector).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FaceBox:
    """Виявлене обличчя: рамка (x, y, w, h), 5 точок і упевненість."""

    bbox: np.ndarray
    landmarks5: np.ndarray
    score: float


def five_from_mesh(face: np.ndarray) -> np.ndarray:
    """5 точок у порядку шаблону ArcFace з 468 точок Face Mesh."""
    face = np.asarray(face, dtype=np.float32)
    return np.stack([
        (face[33] + face[133]) / 2,   # око ліворуч у кадрі
        (face[362] + face[263]) / 2,  # око праворуч у кадрі
        face[1],                      # кінчик носа
        face[61],                     # кутик рота ліворуч у кадрі
        face[291],                    # кутик рота праворуч у кадрі
    ])


class FaceDetector(ABC):
    """Інтерфейс детектора облич."""

    name = "detector"

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        """Список облич у кадрі."""


class YuNetDetector(FaceDetector):
    """Детектор YuNet (cv2.FaceDetectorYN) на зменшеному кадрі."""

    name = "yunet"

    def __init__(self, model_path: str | Path, score_threshold: float = 0.6,
                 max_side: int = 480):
        self.max_side = max_side
        self.det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320),
                                             score_threshold, 0.3, 50)
        self._size = None

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        h, w = frame_bgr.shape[:2]
        k = min(1.0, self.max_side / max(h, w))
        img = cv2.resize(frame_bgr, (round(w * k), round(h * k))) if k < 1 else frame_bgr
        size = (img.shape[1], img.shape[0])
        if size != self._size:
            self.det.setInputSize(size)
            self._size = size
        _, faces = self.det.detect(img)
        result = []
        for f in faces if faces is not None else []:
            result.append(FaceBox(bbox=f[:4] / k, landmarks5=f[4:14].reshape(5, 2) / k,
                                  score=float(f[14])))
        return result


class MeshDetector(FaceDetector):
    """Запасний варіант без моделі: обличчя за точками Face Mesh (упевненість 1)."""

    name = "facemesh"

    def __init__(self, verifier):
        self.verifier = verifier

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        face, _ = self.verifier.detect(frame_bgr)
        if face is None:
            return []
        x0, y0 = face.min(axis=0)
        x1, y1 = face.max(axis=0)
        return [FaceBox(np.array([x0, y0, x1 - x0, y1 - y0]), five_from_mesh(face), 1.0)]


def create_detector(model_path: str | Path | None, verifier=None, min_face: int = 40,
                    max_side: int = 640) -> FaceDetector:
    """MTCNN (каталог з mtcnn_*.onnx), YuNet (файл .onnx) або MeshDetector, якщо моделей немає."""
    if model_path:
        path = Path(model_path)
        from .mtcnn import MTCNN

        if path.is_dir() and MTCNN.available(path):
            return MTCNN(path, min_face=min_face, max_side=max_side)
        if path.is_file():
            return YuNetDetector(path)
    log.warning("модель детектора %s не знайдено, обличчя шукається за Face Mesh "
                "(запустіть scripts/download_models.py)", model_path)
    return MeshDetector(verifier)
