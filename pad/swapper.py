"""Підміна обличчя моделлю inswapper_128 – тією самою, яку використовує Deep-Live-Cam.

Deep-Live-Cam для кожного кадру знаходить обличчя, обчислює ембединг ArcFace
(w600k_r50) цільового обличчя і передає його разом з вирівняним фрагментом
кадру 128 × 128 у мережу inswapper_128, а результат вклеює назад у кадр з
розмитою маскою. Тут той самий алгоритм відтворено без пакета insightface
(за кодом insightface.model_zoo.inswapper, ліцензія MIT для коду; моделі –
лише для некомерційного дослідницького використання), а обличчя і 5 точок
знаходить MTCNN прототипу. Використовується для генерування атак з записаних
відео (scripts/generate_attacks.py), коли немає GPU для підміни в реальному часі.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .embedder import ARCFACE_TEMPLATE

SWAP_FILES = {
    "inswapper_128.onnx": ("https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx",
                           "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"),
    "w600k_r50.onnx": ("https://huggingface.co/hacksider/deep-live-cam/resolve/main/buffalo_l/buffalo_l/w600k_r50.onnx",
                       "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"),
}


def _session(path: Path, providers: list[str] | None, threads: int | None = None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = threads or os.cpu_count() or 4
    available = ort.get_available_providers()
    providers = [p for p in (providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]) if p in available]
    return ort.InferenceSession(str(path), opts, providers=providers)


def norm_crop_matrix(landmarks5: np.ndarray, size: int) -> np.ndarray:
    """Матриця вирівнювання до шаблону ArcFace, як insightface.utils.face_align.estimate_norm."""
    if size % 112 == 0:
        ratio, diff_x = size / 112.0, 0.0
    else:
        ratio, diff_x = size / 128.0, 8.0 * size / 128.0
    dst = ARCFACE_TEMPLATE * ratio
    dst[:, 0] += diff_x
    m, _ = cv2.estimateAffinePartial2D(np.asarray(landmarks5, np.float32), dst, method=cv2.LMEDS)
    return m


class ArcFace:
    """Ембединг ArcFace w600k_r50 (512) для джерела ідентичності inswapper."""

    def __init__(self, path: str | Path, providers: list[str] | None = None, threads: int | None = None):
        self.session = _session(Path(path), providers, threads)
        self.input = self.session.get_inputs()[0].name

    def embed(self, image_bgr: np.ndarray, landmarks5: np.ndarray) -> np.ndarray:
        m = norm_crop_matrix(landmarks5, 112)
        crop = cv2.warpAffine(image_bgr, m, (112, 112), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        e = self.session.run(None, {self.input: blob})[0].reshape(-1)
        return e / np.linalg.norm(e)


class InSwapper:
    """Мережа inswapper_128 з вклеюванням результату в кадр."""

    size = 128

    def __init__(self, path: str | Path, providers: list[str] | None = None, threads: int | None = None):
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(path))
        self.emap = numpy_helper.to_array(model.graph.initializer[-1])
        self.session = _session(Path(path), providers, threads)
        self.inputs = [i.name for i in self.session.get_inputs()]

    def latent(self, source_embedding: np.ndarray) -> np.ndarray:
        lat = source_embedding.reshape(1, -1) @ self.emap
        return (lat / np.linalg.norm(lat)).astype(np.float32)

    def swap(self, frame_bgr: np.ndarray, landmarks5: np.ndarray, latent: np.ndarray) -> np.ndarray:
        m = norm_crop_matrix(landmarks5, self.size)
        aimg = cv2.warpAffine(frame_bgr, m, (self.size, self.size), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / 255.0, (self.size, self.size), (0.0, 0.0, 0.0), swapRB=True)
        pred = self.session.run(None, {self.inputs[0]: blob, self.inputs[1]: latent})[0]
        fake = np.clip(255 * pred.transpose(0, 2, 3, 1)[0], 0, 255).astype(np.uint8)[:, :, ::-1]
        return paste_back(frame_bgr, aimg, fake, m)


def paste_back(frame: np.ndarray, aimg: np.ndarray, fake: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Вклеює згенероване обличчя в кадр з еродованою і розмитою маскою (як у insightface)."""
    h, w = frame.shape[:2]
    im = cv2.invertAffineTransform(m)
    white = np.full(aimg.shape[:2], 255, np.float32)
    fake_full = cv2.warpAffine(fake, im, (w, h), borderValue=0.0)
    mask = cv2.warpAffine(white, im, (w, h), borderValue=0.0)
    mask[mask > 20] = 255
    ys, xs = np.where(mask == 255)
    if len(ys) == 0:
        return frame
    size = int(np.sqrt((ys.max() - ys.min()) * (xs.max() - xs.min())))
    k = max(size // 10, 10)
    mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
    k = max(size // 20, 5)
    mask = cv2.GaussianBlur(mask, (2 * k + 1, 2 * k + 1), 0)[:, :, None] / 255.0
    return (mask * fake_full + (1 - mask) * frame.astype(np.float32)).astype(np.uint8)


def ensure_models(model_dir: str | Path) -> Path:
    """Завантажує inswapper_128 і w600k_r50 (≈ 730 МБ), перевіряє SHA-256."""
    import hashlib
    import urllib.request

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in SWAP_FILES.items():
        dst = model_dir / name
        if not dst.exists():
            print(f"завантаження {name} ...")
            tmp = dst.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dst)
        h = hashlib.sha256()
        with open(dst, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            raise RuntimeError(f"хеш {name} не збігається, видаліть файл і завантажте знову")
    return model_dir
