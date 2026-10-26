"""Спрощений конвеєр кадру: детекція, ключові точки, ембединг -> FrameData.

У повній системі ці дані надає конвеєр розпізнавання (MTCNN + KCF +
MobileFaceNet), а модуль виявлення атак лише повторно використовує ембединг.
Якщо детектор не знаходить обличчя (наприклад, його закрито рукою),
ембединг обчислюється за точками Face Mesh або за останніми відомими точками,
бо саме зміна ембединга під час перекриття і є ознакою атаки.
"""

from __future__ import annotations

import time

import numpy as np

from .detector import FaceBox, FaceDetector, five_from_mesh
from .embedder import FaceEmbedder, align_face
from .features import l2_normalize, reference_embedding
from .structures import FrameData
from .verifier import ChallengeVerifier


class FramePipeline:
    """Обробляє кадр і вимірює час кожного етапу."""

    def __init__(self, detector: FaceDetector, embedder: FaceEmbedder,
                 verifier: ChallengeVerifier, min_det_score: float = 0.6,
                 face_hold_s: float = 1.0):
        self.detector = detector
        self.embedder = embedder
        self.verifier = verifier
        self.min_det_score = min_det_score
        self.face_hold_s = face_hold_s
        self._last_five: tuple[float, np.ndarray] | None = None
        self.last_box: FaceBox | None = None

    def warmup(self, size: tuple[int, int] = (1280, 720)) -> None:
        """Перший прохід усіх моделей, щоб їх ініціалізація не потрапила у вимірювання."""
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        self.process(frame, -10.0, landmarks=True, hands=True)
        self.embedder.embed(np.zeros((self.embedder.input_size, self.embedder.input_size, 3), np.uint8))
        self._last_five = None

    def process(self, frame_bgr: np.ndarray, t: float, landmarks: bool = True,
                hands: bool = True) -> tuple[FrameData, dict]:
        """Повертає дані кадру і час етапів у мілісекундах."""
        timings = {}
        t0 = time.perf_counter()
        faces = self.detector.detect(frame_bgr)
        box = max(faces, key=lambda f: f.bbox[2] * f.bbox[3]) if faces else None
        self.last_box = box
        t1 = time.perf_counter()
        timings["detect_ms"] = (t1 - t0) * 1000

        face = hand = None
        found: list[np.ndarray] = []
        if landmarks:
            face, found = self.verifier.detect_all(frame_bgr, hands=hands)
            hand = self.verifier.closest_hand(found, face)
        t2 = time.perf_counter()
        timings["landmarks_ms"] = (t2 - t1) * 1000

        score = box.score if box is not None else 0.0
        if box is not None and score >= self.min_det_score:
            five = box.landmarks5
        elif face is not None:
            five = five_from_mesh(face)
        elif self._last_five is not None and t - self._last_five[0] <= self.face_hold_s:
            five = self._last_five[1]
        else:
            five = None
        embedding = None
        if five is not None:
            self._last_five = (t, five)
            embedding = self.embedder.embed(align_face(frame_bgr, five, self.embedder.input_size))
        t3 = time.perf_counter()
        timings["embed_ms"] = (t3 - t2) * 1000
        timings["total_ms"] = (t3 - t0) * 1000

        h, w = frame_bgr.shape[:2]
        return FrameData(t=t, embedding=embedding, det_score=score, face=face, hand=hand,
                         hands=found or None, size=(w, h)), timings

    def faces(self, frame_bgr: np.ndarray, min_score: float = 0.9,
              max_faces: int = 5) -> tuple[list[tuple[FaceBox, np.ndarray]], dict]:
        """Усі обличчя кадру з ембедингами (для моніторингу) і час етапів у мілісекундах.

        Face Mesh і Hands тут не потрібні: кожне обличчя вирівнюється за 5 точками
        детектора. Беруться max_faces найбільших облич з упевненістю >= min_score.
        """
        t0 = time.perf_counter()
        boxes = [f for f in self.detector.detect(frame_bgr) if f.score >= min_score]
        boxes = sorted(boxes, key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)[:max_faces]
        t1 = time.perf_counter()
        result = [(b, self.embedder.embed(align_face(frame_bgr, b.landmarks5, self.embedder.input_size)))
                  for b in boxes]
        t2 = time.perf_counter()
        return result, {"detect_ms": (t1 - t0) * 1000, "landmarks_ms": 0.0, "embed_ms": (t2 - t1) * 1000,
                        "total_ms": (t2 - t0) * 1000}

    def embed_image(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Ембединг обличчя з фотографії (наприклад, цільового обличчя для атаки) або None.

        Береться найупевненіше обличчя детектора, а якщо його немає – точки Face Mesh.
        """
        faces = self.detector.detect(image_bgr)
        if faces:
            five = max(faces, key=lambda f: f.score).landmarks5
        else:
            face, _ = self.verifier.detect(image_bgr, hands=False)
            if face is None:
                return None
            five = five_from_mesh(face)
        return self.embedder.embed(align_face(image_bgr, five, self.embedder.input_size))


def enroll_embedding(frames: list[FrameData], min_score: float = 0.9) -> np.ndarray | None:
    """Ембединг заявленої особи як нормоване середнє кадрів з упевненістю >= min_score."""
    ref = reference_embedding([f.embedding for f in frames], [f.det_score for f in frames], min_score)
    return l2_normalize(ref) if ref is not None else None
