"""Збирання компонентів прототипу за конфігураційним файлом configs/pad.yaml."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .classifiers import AttackClassifier, ThresholdClassifier, load_classifier
from .detector import create_detector
from .embedder import create_embedder
from .pipeline import FramePipeline
from .session import SessionManager
from .settings import PadSettings
from .verifier import ChallengeVerifier

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | None) -> Path | None:
    """Шлях відносно кореня проєкту."""
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path | None = None) -> dict:
    """Читає YAML-конфігурацію (за замовчуванням configs/pad.yaml)."""
    path = Path(path) if path else ROOT / "configs" / "pad.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def default_classifier() -> AttackClassifier:
    """Базовий поріг на мінімумі s_ref, поки модель не навчено на пілотному наборі."""
    clf = ThresholdClassifier("s_ref_during_min", threshold=0.3, scale=0.05)
    clf.version = "threshold-default"
    return clf


def build_classifier(cfg: dict, override: str | None = None) -> AttackClassifier:
    """Класифікатор з файлу (override або cfg['classifier']) чи базовий поріг."""
    path = resolve(override or cfg.get("classifier"))
    if path is not None and path.exists():
        clf = load_classifier(path)
        log.info("класифікатор %s (%s)", path.name, clf.version)
        return clf
    log.warning("файл класифікатора %s не знайдено, використовується базовий поріг "
                "s_ref_during_min < 0,3 (до навчання на пілотному наборі)", path)
    return default_classifier()


def build_runtime(cfg: dict, classifier: str | None = None) -> tuple[FramePipeline, SessionManager]:
    """Створює конвеєр кадру і SessionManager."""
    settings = PadSettings.from_dict(cfg.get("pad", {}))
    models = cfg.get("models", {})
    verifier = ChallengeVerifier(mirrored=False)
    detector = create_detector(resolve(models.get("detector")), verifier,
                               models.get("detector_min_face", 40), models.get("detector_max_side", 640))
    embedder = create_embedder(resolve(models.get("embedder")), models.get("embedder_profile", "mobilefacenet"),
                               models.get("embedder_threads", 4))
    pipeline = FramePipeline(detector, embedder, verifier, face_hold_s=settings.face_hold_s)
    manager = SessionManager(build_classifier(cfg, classifier), settings, verifier=verifier)
    log.info("детектор: %s, ембединг: %s", detector.name, embedder.name)
    return pipeline, manager
