"""MTCNN і MobileFaceNet: допоміжні функції каскаду, вибір детектора, попередня обробка."""

from pathlib import Path

import numpy as np
import pytest

from pad.detector import MeshDetector, YuNetDetector, create_detector
from pad.embedder import PROFILES, create_embedder, preprocess
from pad.mtcnn import MTCNN, bbreg, nms, rerec

MODELS = Path(__file__).resolve().parents[1] / "models"
HAS_MTCNN = MTCNN.available(MODELS)
HAS_MFN = (MODELS / "mobilefacenet_int8.onnx").exists()


def test_nms_removes_overlapping_box():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [200, 200, 260, 260]], float)
    scores = np.array([0.9, 0.8, 0.7])
    assert list(nms(boxes, scores, 0.5)) == [0, 2]


def test_nms_min_mode_suppresses_nested_box():
    boxes = np.array([[0, 0, 100, 100], [20, 20, 50, 50]], float)
    scores = np.array([0.9, 0.95])
    assert list(nms(boxes, scores, 0.7, mode="min")) == [1]
    assert len(nms(boxes, scores, 0.7, mode="union")) == 2


def test_nms_empty():
    assert len(nms(np.zeros((0, 4)), np.zeros(0), 0.5)) == 0


def test_rerec_makes_square_with_same_center():
    b = rerec(np.array([[10.0, 20.0, 50.0, 100.0, 0.9]]))
    w, h = b[0, 2] - b[0, 0], b[0, 3] - b[0, 1]
    assert w == pytest.approx(h)
    assert (b[0, 0] + b[0, 2]) / 2 == pytest.approx(30.0)
    assert (b[0, 1] + b[0, 3]) / 2 == pytest.approx(60.0)


def test_bbreg_shifts_by_relative_offsets():
    b = bbreg(np.array([[0.0, 0.0, 9.0, 9.0, 1.0]]), np.array([[0.1, 0.0, -0.1, 0.2]]))
    assert b[0, :4] == pytest.approx([1.0, 0.0, 8.0, 11.0])


def test_mobilefacenet_profile_uses_imagenet_normalization():
    face = np.full((112, 112, 3), (103, 116, 123), np.uint8)  # BGR ≈ середнє ImageNet
    x = preprocess(face, PROFILES["mobilefacenet"])
    assert x.shape == (1, 3, 112, 112)
    assert np.abs(x).max() < 0.02


def test_create_detector_falls_back_without_models(tmp_path):
    assert isinstance(create_detector(tmp_path, verifier=None), MeshDetector)


@pytest.mark.skipif(not HAS_MTCNN, reason="немає mtcnn_*.onnx (scripts/export_models.py)")
def test_create_detector_picks_mtcnn_for_model_dir():
    assert isinstance(create_detector(MODELS), MTCNN)
    yunet = MODELS / "face_detection_yunet_2023mar.onnx"
    if yunet.exists():
        assert isinstance(create_detector(yunet), YuNetDetector)


@pytest.mark.skipif(not HAS_MTCNN, reason="немає mtcnn_*.onnx (scripts/export_models.py)")
def test_mtcnn_finds_no_face_on_uniform_frame():
    det = MTCNN(MODELS)
    assert det.detect(np.full((480, 640, 3), 128, np.uint8)) == []


@pytest.mark.skipif(not (HAS_MTCNN and HAS_MFN), reason="немає моделей MTCNN / MobileFaceNet")
def test_mobilefacenet_int8_embedding_is_normalized_128d():
    emb = create_embedder(MODELS / "mobilefacenet_int8.onnx", "mobilefacenet")
    rng = np.random.default_rng(0)
    e1 = emb.embed(rng.integers(0, 255, (112, 112, 3), dtype=np.uint8))
    e2 = emb.embed(rng.integers(0, 255, (112, 112, 3), dtype=np.uint8))
    assert e1.shape == (128,)
    assert np.linalg.norm(e1) == pytest.approx(1.0, abs=1e-5)
    assert float(e1 @ e2) < 0.999
