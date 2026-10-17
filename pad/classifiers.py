"""Класифікатори атак (інтерфейс AttackClassifier, шаблон «Стратегія»).

ThresholdClassifier – базовий варіант: поріг на одній ознаці (мінімум s_ref
під час виклику). LogisticClassifier – логістична регресія на 26 ознаках,
обрана для курсової роботи: оцінка атаки σ(w · x + b) у діапазоні [0; 1].
Оцінка сеансу – максимум оцінок викликів (session_score).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


class AttackClassifier(ABC):
    """Інтерфейс класифікатора: ознаки виклику -> оцінка атаки в [0; 1]."""

    kind = ""
    version = "untrained"

    @abstractmethod
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Оцінки атаки для матриці ознак (n, 26)."""

    def predict(self, x: np.ndarray) -> float:
        """Оцінка атаки для одного вектора ознак."""
        return float(self.predict_batch(np.asarray(x, dtype=float).reshape(1, -1))[0])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AttackClassifier":
        """Навчає класифікатор (y = 1 – атака, 0 – справжнє подання)."""
        return self

    def to_dict(self) -> dict:
        """Параметри моделі для збереження у JSON."""
        return {"kind": self.kind, "version": self.version}

    def save(self, path: str | Path) -> None:
        """Зберігає модель у JSON-файл."""
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def session_score(challenge_scores) -> float:
    """Оцінка сеансу – найбільша з оцінок викликів."""
    return float(np.max(np.asarray(challenge_scores, dtype=float)))


class ThresholdClassifier(AttackClassifier):
    """Поріг на одній ознаці: чим менше значення ознаки, тим вища оцінка атаки.

    Оцінка σ((threshold – x) / scale) дорівнює 0,5 саме на порозі.
    """

    kind = "threshold"

    def __init__(self, feature: str = "s_ref_during_min", threshold: float = 0.6,
                 scale: float = 0.05, names: list[str] | None = None):
        self.feature = feature
        self.threshold = float(threshold)
        self.scale = float(scale)
        self.names = list(names or FEATURE_NAMES)
        self.index = self.names.index(feature)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=float)[:, self.index]
        return _sigmoid((self.threshold - x) / max(self.scale, 1e-9))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ThresholdClassifier":
        """Обирає поріг з найбільшою збалансованою точністю на навчальних даних."""
        x = np.asarray(X, dtype=float)[:, self.index]
        y = np.asarray(y).astype(int)
        best = (-1.0, self.threshold)
        for thr in np.unique(x):
            pred = x <= thr
            tpr = pred[y == 1].mean() if (y == 1).any() else 0.0
            tnr = (~pred[y == 0]).mean() if (y == 0).any() else 0.0
            if (tpr + tnr) / 2 > best[0]:
                best = ((tpr + tnr) / 2, float(thr))
        self.threshold = best[1]
        self.scale = max(float(np.std(x)) * 0.1, 1e-3)
        self.version = f"threshold-{self.feature}"
        return self

    def to_dict(self) -> dict:
        return {**super().to_dict(), "feature": self.feature, "threshold": self.threshold,
                "scale": self.scale, "names": self.names}

    @classmethod
    def from_dict(cls, d: dict) -> "ThresholdClassifier":
        obj = cls(d["feature"], d["threshold"], d["scale"], d.get("names"))
        obj.version = d.get("version", obj.version)
        return obj


class LogisticClassifier(AttackClassifier):
    """Логістична регресія: StandardScaler + LogisticRegression (L2, class_weight='balanced').

    Після навчання стандартизацію згорнуто у ваги, тому для інференсу
    достатньо weights і bias: p = σ(w · x + b).
    """

    kind = "logistic"

    def __init__(self, C: float = 1.0, seed: int = 0, names: list[str] | None = None):
        self.C = C
        self.seed = seed
        self.names = list(names or FEATURE_NAMES)
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_std_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticClassifier":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000,
                               random_state=self.seed),
        )
        pipe.fit(np.asarray(X, dtype=float), np.asarray(y).astype(int))
        scaler, lr = pipe[0], pipe[1]
        scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
        self.mean_, self.scale_ = scaler.mean_.copy(), scale
        self.coef_std_ = lr.coef_[0].copy()
        self.weights = self.coef_std_ / scale
        self.bias = float(lr.intercept_[0] - np.sum(self.coef_std_ * self.mean_ / scale))
        self.version = f"logistic-C{self.C}"
        return self

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("модель не навчено")
        return _sigmoid(np.asarray(X, dtype=float) @ self.weights + self.bias)

    def importance(self) -> list[tuple[str, float]]:
        """Ваги ознак у стандартизованій шкалі, впорядковані за модулем."""
        return sorted(zip(self.names, map(float, self.coef_std_)), key=lambda p: -abs(p[1]))

    def to_dict(self) -> dict:
        return {**super().to_dict(), "C": self.C, "names": self.names,
                "weights": self.weights.tolist(), "bias": self.bias,
                "mean": self.mean_.tolist(), "scale": self.scale_.tolist(),
                "coef_std": self.coef_std_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticClassifier":
        obj = cls(d.get("C", 1.0), names=d.get("names"))
        obj.weights = np.array(d["weights"], dtype=float)
        obj.bias = float(d["bias"])
        obj.mean_ = np.array(d.get("mean", np.zeros_like(obj.weights)))
        obj.scale_ = np.array(d.get("scale", np.ones_like(obj.weights)))
        obj.coef_std_ = np.array(d.get("coef_std", obj.weights * obj.scale_))
        obj.version = d.get("version", obj.version)
        return obj

    def export_onnx(self, path: str | Path) -> None:
        """Експортує модель у формат ONNX: score = Sigmoid(X · w + b)."""
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        w = numpy_helper.from_array(self.weights.astype(np.float32).reshape(-1, 1), "w")
        b = numpy_helper.from_array(np.array([self.bias], dtype=np.float32), "b")
        nodes = [helper.make_node("MatMul", ["features", "w"], ["z0"]),
                 helper.make_node("Add", ["z0", "b"], ["z"]),
                 helper.make_node("Sigmoid", ["z"], ["score"])]
        graph = helper.make_graph(
            nodes, "pad_logistic",
            [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, len(self.names)])],
            [helper.make_tensor_value_info("score", TensorProto.FLOAT, [None, 1])],
            initializer=[w, b])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                                  producer_name="frs-pad")
        model.ir_version = 8
        model.metadata_props.add(key="version", value=self.version)
        model.metadata_props.add(key="feature_names", value=",".join(self.names))
        onnx.checker.check_model(model)
        onnx.save(model, str(path))


class OnnxClassifier(AttackClassifier):
    """Класифікатор, завантажений з ONNX-файлу (вхід features, вихід score)."""

    kind = "onnx"

    def __init__(self, path: str | Path):
        import onnxruntime as ort

        self.path = str(path)
        self.session = ort.InferenceSession(self.path, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.version = meta.get("version", Path(path).stem)
        self.names = meta.get("feature_names", ",".join(FEATURE_NAMES)).split(",")
        self.input_name = self.session.get_inputs()[0].name

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        out = self.session.run(None, {self.input_name: np.asarray(X, dtype=np.float32)})[0]
        return np.asarray(out, dtype=float).reshape(-1)


def load_classifier(path: str | Path) -> AttackClassifier:
    """Завантажує класифікатор з файлу .json або .onnx."""
    path = Path(path)
    if path.suffix.lower() == ".onnx":
        return OnnxClassifier(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    if d["kind"] == "threshold":
        return ThresholdClassifier.from_dict(d)
    if d["kind"] == "logistic":
        return LogisticClassifier.from_dict(d)
    raise ValueError(f"невідомий тип моделі: {d['kind']}")
