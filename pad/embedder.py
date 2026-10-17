"""Адаптер екстрактора ембедингів з інтерфейсом embed(face_crop) -> ndarray[128].

Основна модель – MobileFaceNet з INT8-квантуванням (ONNX, scripts/export_models.py):
вирівняне обличчя 112 × 112 -> 128-вимірний вектор. Модель SFace з OpenCV Zoo
лишається запасним варіантом (профіль sface). Якщо файлу моделі немає,
використовується заглушка StubEmbedder (див. її опис).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from .features import l2_normalize

log = logging.getLogger(__name__)

#: Еталонні положення 5 точок (очі, ніс, кутики рота) на обличчі 112 × 112 (ArcFace).
ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

#: Профілі попередньої обробки: середнє і масштаб (за каналами RGB), порядок каналів.
#: MobileFaceNet (foamliu) навчена з нормалізацією ImageNet, як у torchvision.
PROFILES = {
    "sface": {"mean": 0.0, "std": 1.0, "rgb": True},
    "mobilefacenet": {"mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375], "rgb": True},
}


def preprocess(face_crop: np.ndarray, prep: dict, size: int = 112) -> np.ndarray:
    """Обличчя BGR -> тензор 1 × 3 × size × size для мережі."""
    img = face_crop
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size))
    if prep["rgb"]:
        img = img[:, :, ::-1]
    mean = np.asarray(prep["mean"], dtype=np.float32)
    std = np.asarray(prep["std"], dtype=np.float32)
    x = (img.astype(np.float32) - mean) / std
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def align_face(frame_bgr: np.ndarray, landmarks5: np.ndarray, size: int = 112) -> np.ndarray:
    """Вирівнює обличчя подібним перетворенням за 5 точками до шаблону ArcFace."""
    src = np.asarray(landmarks5, dtype=np.float32).reshape(5, 2)
    dst = ARCFACE_TEMPLATE * (size / 112.0)
    m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if m is None:
        m = cv2.getAffineTransform(src[:3], dst[:3])
    return cv2.warpAffine(frame_bgr, m, (size, size), borderValue=0)


class FaceEmbedder(ABC):
    """Інтерфейс екстрактора: вирівняне обличчя -> L2-нормований вектор."""

    dim = 128
    name = "embedder"
    input_size = 112

    @abstractmethod
    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        """Ембединг вирівняного обличчя (BGR, input_size × input_size)."""


class OnnxFaceEmbedder(FaceEmbedder):
    """Ембединг засобами ONNX Runtime (MobileFaceNet INT8 або запасна SFace)."""

    def __init__(self, model_path: str | Path, profile: str = "sface", input_size: int = 112,
                 threads: int = 4):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = threads
        # без активного очікування потоків, щоб не забирати процесор у MediaPipe між кадрами
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.dim = int(self.session.get_outputs()[0].shape[-1])
        self.prep = PROFILES[profile]
        self.input_size = input_size
        self.name = f"{Path(model_path).stem} ({profile})"

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        x = preprocess(face_crop, self.prep, self.input_size)
        out = self.session.run(None, {self.input_name: x})[0]
        return l2_normalize(out.reshape(-1))


class StubEmbedder(FaceEmbedder):
    """Заглушка без нейромережі: випадкова проєкція нормованого сірого обличчя 32 × 32.

    Вектор змінюється, коли обличчя закривають рукою або повертають голову,
    тому на заглушці можна перевірити роботу інтерфейсу і графіка s_ref(t).
    Ідентичність людини він не розрізняє, тому s_enr і результати
    класифікації на заглушці не мають сенсу.
    """

    name = "stub"

    def __init__(self, dim: int = 128, seed: int = 0):
        self.dim = dim
        self.proj = np.random.default_rng(seed).normal(size=(32 * 32, dim)).astype(np.float32)

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
        x = cv2.resize(gray, (32, 32)).astype(np.float32).reshape(-1)
        x = (x - x.mean()) / (x.std() + 1e-6)
        return l2_normalize(x @ self.proj)


def create_embedder(model_path: str | Path | None, profile: str = "mobilefacenet",
                    threads: int = 4) -> FaceEmbedder:
    """ONNX-екстрактор, якщо файл моделі є, інакше заглушка з попередженням."""
    if model_path and Path(model_path).exists():
        return OnnxFaceEmbedder(model_path, profile, threads=threads)
    log.warning("модель ембедингів %s не знайдено, використовується заглушка StubEmbedder "
                "(запустіть scripts/download_models.py)", model_path)
    return StubEmbedder()
