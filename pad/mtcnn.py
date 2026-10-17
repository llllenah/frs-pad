"""Детектор облич MTCNN (P-Net, R-Net, O-Net) у форматі ONNX, виконання через ONNX Runtime.

Каскад повторює алгоритм Zhang et al. [10] у реалізації facenet-pytorch (MIT):
піраміда масштабів для P-Net, придушення немаксимумів (NMS), уточнення рамок
R-Net та O-Net і 5 ключових точок з O-Net. Як і в пояснювальній записці
(підрозділ 1.2.2), кадр перед детекцією зменшується до max_side пікселів за
довшою стороною, а мінімальний розмір обличчя задається min_face – це
скорочує кількість масштабів піраміди і час детекції на CPU.

Моделі створює scripts/export_models.py: mtcnn_pnet.onnx, mtcnn_rnet.onnx, mtcnn_onet.onnx.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .detector import FaceBox, FaceDetector


def _session(path: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = threads
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float, mode: str = "union") -> np.ndarray:
    """Індекси рамок після придушення немаксимумів (union – IoU, min – перетин до меншої площі)."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        if mode == "min":
            ovr = inter / np.minimum(area[i], area[order[1:]])
        else:
            ovr = inter / (area[i] + area[order[1:]] - inter)
        order = order[1:][ovr <= threshold]
    return np.array(keep, dtype=int)


def rerec(boxes: np.ndarray) -> np.ndarray:
    """Перетворює рамки на квадрати з тим самим центром."""
    boxes = boxes.copy()
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    side = np.maximum(w, h)
    boxes[:, 0] = boxes[:, 0] + w * 0.5 - side * 0.5
    boxes[:, 1] = boxes[:, 1] + h * 0.5 - side * 0.5
    boxes[:, 2] = boxes[:, 0] + side
    boxes[:, 3] = boxes[:, 1] + side
    return boxes


def bbreg(boxes: np.ndarray, reg: np.ndarray) -> np.ndarray:
    """Уточнення рамок за регресією мережі."""
    boxes = boxes.copy()
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    boxes[:, 0] += reg[:, 0] * w
    boxes[:, 1] += reg[:, 1] * h
    boxes[:, 2] += reg[:, 2] * w
    boxes[:, 3] += reg[:, 3] * h
    return boxes


def _normalize(img_rgb: np.ndarray) -> np.ndarray:
    return ((img_rgb.astype(np.float32) - 127.5) * 0.0078125).transpose(2, 0, 1)


def _crops(img_rgb: np.ndarray, boxes: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Вирізані й масштабовані фрагменти; повертає також маску рамок з непорожнім фрагментом."""
    h, w = img_rgb.shape[:2]
    out, ok = [], []
    for b in boxes:
        x1, y1 = max(int(b[0]), 1), max(int(b[1]), 1)
        x2, y2 = min(int(b[2]), w), min(int(b[3]), h)
        if x2 <= x1 - 1 or y2 <= y1 - 1:
            ok.append(False)
            continue
        crop = img_rgb[y1 - 1:y2, x1 - 1:x2]
        if crop.size == 0:
            ok.append(False)
            continue
        out.append(_normalize(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)))
        ok.append(True)
    arr = np.stack(out).astype(np.float32) if out else np.zeros((0, 3, size, size), np.float32)
    return arr, np.array(ok, dtype=bool)


class MTCNN(FaceDetector):
    """Каскад MTCNN на ONNX Runtime з попереднім масштабуванням кадру."""

    name = "mtcnn"

    def __init__(self, model_dir: str | Path, min_face: int = 40, max_side: int = 640,
                 thresholds: tuple[float, float, float] = (0.6, 0.7, 0.7), factor: float = 0.709,
                 threads: int = 2):
        model_dir = Path(model_dir)
        self.pnet = _session(model_dir / "mtcnn_pnet.onnx", threads)
        self.rnet = _session(model_dir / "mtcnn_rnet.onnx", threads)
        self.onet = _session(model_dir / "mtcnn_onet.onnx", threads)
        self.min_face = min_face
        self.max_side = max_side
        self.thresholds = thresholds
        self.factor = factor

    @staticmethod
    def available(model_dir: str | Path) -> bool:
        model_dir = Path(model_dir)
        return all((model_dir / f"mtcnn_{n}.onnx").exists() for n in ("pnet", "rnet", "onet"))

    # ------------------------------------------------------------------ етапи
    def _stage1(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        m = 12.0 / self.min_face
        minl = min(h, w) * m
        scales = []
        while minl >= 12:
            scales.append(m * self.factor ** len(scales))
            minl *= self.factor
        all_boxes = []
        for scale in scales:
            hs, ws = int(h * scale + 1), int(w * scale + 1)
            im = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
            reg, prob = self.pnet.run(None, {"input": _normalize(im)[None]})
            prob = prob[0, 1]
            ys, xs = np.nonzero(prob >= self.thresholds[0])
            if len(ys) == 0:
                continue
            score = prob[ys, xs]
            r = reg[0][:, ys, xs].T
            q1x = np.floor((2 * xs + 1) / scale)
            q1y = np.floor((2 * ys + 1) / scale)
            q2x = np.floor((2 * xs + 12) / scale)
            q2y = np.floor((2 * ys + 12) / scale)
            boxes = np.column_stack([q1x, q1y, q2x, q2y, score, r])
            boxes = boxes[nms(boxes[:, :4], boxes[:, 4], 0.5)]
            all_boxes.append(boxes)
        if not all_boxes:
            return np.zeros((0, 5))
        boxes = np.vstack(all_boxes)
        boxes = boxes[nms(boxes[:, :4], boxes[:, 4], 0.7)]
        bw = boxes[:, 2] - boxes[:, 0]
        bh = boxes[:, 3] - boxes[:, 1]
        out = np.column_stack([boxes[:, 0] + boxes[:, 5] * bw, boxes[:, 1] + boxes[:, 6] * bh,
                               boxes[:, 2] + boxes[:, 7] * bw, boxes[:, 3] + boxes[:, 8] * bh, boxes[:, 4]])
        return rerec(out)

    def _stage2(self, img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        crops, ok = _crops(img, boxes, 24)
        boxes = boxes[ok]
        if len(boxes) == 0:
            return boxes
        reg, prob = self.rnet.run(None, {"input": crops})
        keep = prob[:, 1] > self.thresholds[1]
        boxes = np.column_stack([boxes[keep, :4], prob[keep, 1]])
        reg = reg[keep]
        idx = nms(boxes[:, :4], boxes[:, 4], 0.7)
        return rerec(bbreg(boxes[idx], reg[idx]))

    def _stage3(self, img: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        crops, ok = _crops(img, boxes, 48)
        boxes = boxes[ok]
        if len(boxes) == 0:
            return boxes, np.zeros((0, 5, 2))
        reg, pts, prob = self.onet.run(None, {"input": crops})
        keep = prob[:, 1] > self.thresholds[2]
        boxes = np.column_stack([boxes[keep, :4], prob[keep, 1]])
        reg, pts = reg[keep], pts[keep]
        w = boxes[:, 2] - boxes[:, 0] + 1
        h = boxes[:, 3] - boxes[:, 1] + 1
        px = w[:, None] * pts[:, :5] + boxes[:, 0:1] - 1
        py = h[:, None] * pts[:, 5:10] + boxes[:, 1:2] - 1
        points = np.stack([px, py], axis=2)
        boxes = bbreg(boxes, reg)
        idx = nms(boxes[:, :4], boxes[:, 4], 0.7, mode="min")
        return boxes[idx], points[idx]

    # ------------------------------------------------------------------ інтерфейс
    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        h, w = frame_bgr.shape[:2]
        k = min(1.0, self.max_side / max(h, w))
        img = cv2.resize(frame_bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame_bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = self._stage1(img)
        if len(boxes):
            boxes = self._stage2(img, boxes)
        points = np.zeros((0, 5, 2))
        if len(boxes):
            boxes, points = self._stage3(img, boxes)
        result = []
        for b, p in zip(boxes, points):
            x1, y1, x2, y2 = b[:4] / k
            result.append(FaceBox(bbox=np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32),
                                  landmarks5=(p / k).astype(np.float32), score=float(b[4])))
        return result
