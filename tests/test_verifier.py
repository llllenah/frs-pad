"""Тести перевірки викликів на синтетичних ключових точках (FR-54 – FR-56)."""

import numpy as np
import pytest

from pad.challenges import HandSwipeChallenge, HeadTurnChallenge, OcclusionChallenge
from pad.geometry import clip_convex, convex_hull, polygon_area
from pad.structures import FrameData
from pad.synthetic import hand_over_face, synthetic_face, synthetic_hand
from pad.verifier import ChallengeVerifier

SIZE = (1280, 720)


# ----- геометрія -----

def test_hull_and_area_of_square():
    pts = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]], float)
    hull = convex_hull(pts)
    assert len(hull) == 4
    assert polygon_area(hull) == pytest.approx(4.0)


def test_clip_two_squares():
    a = convex_hull(np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float))
    b = convex_hull(np.array([[1, 1], [3, 1], [3, 3], [1, 3]], float))
    assert polygon_area(clip_convex(a, b)) == pytest.approx(1.0)
    c = convex_hull(np.array([[5, 5], [6, 5], [6, 6]], float))
    assert polygon_area(clip_convex(a, c)) == 0.0


# ----- частка перекриття, формула (3.3) -----

def test_coverage_without_hand_is_zero(verifier):
    assert verifier.coverage(synthetic_face(), None) == 0.0


def test_coverage_hand_far_from_face(verifier):
    hand = synthetic_hand((100, 100), 80, 120)
    assert verifier.coverage(synthetic_face(), hand) == 0.0


def test_coverage_whole_face(verifier):
    face = synthetic_face()
    big = synthetic_hand(tuple(face.mean(axis=0)), 2000, 2000)
    assert verifier.coverage(face, big) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("side", ["left", "right"])
def test_cover_half_of_face(verifier, side):
    face = synthetic_face()
    info = verifier.coverage_info(face, hand_over_face(face, side, scale=1.3))
    assert info["coverage"] >= 0.4
    assert info["side"] == side
    small = verifier.coverage_info(face, hand_over_face(face, side, scale=0.6))
    assert small["coverage"] < 0.4


def test_left_half_is_on_image_right_in_raw_frame(verifier):
    face = synthetic_face()
    hand = hand_over_face(face, "left", 1.3)
    assert hand[:, 0].mean() > face[1, 0]  # центр кисті праворуч від носа в кадрі
    assert ChallengeVerifier(mirrored=True).coverage_info(face, hand)["side"] == "right"


def test_cover_mouth_is_lower(verifier):
    face = synthetic_face()
    info = verifier.coverage_info(face, hand_over_face(face, "mouth", 1.3))
    assert info["lower"] and info["coverage"] >= 0.4
    assert not verifier.coverage_info(face, hand_over_face(face, "left", 1.3))["lower"]


def _frames(face, hands, t0=0.0, fps=30.0):
    return [FrameData(t=t0 + i / fps, face=face, hand=h, size=SIZE) for i, h in enumerate(hands)]


def test_occlusion_needs_five_consecutive_frames(verifier):
    face = synthetic_face()
    ok_hand = hand_over_face(face, "left", 1.3)
    ch = OcclusionChallenge("left")
    for f in _frames(face, [ok_hand] * 4):
        ch.add_frame(f, verifier)
    assert not ch.is_performed(verifier)
    ch.add_frame(FrameData(t=0.2, face=face, hand=ok_hand, size=SIZE), verifier)
    assert ch.is_performed(verifier)
    assert ch.performed_at == pytest.approx(0.2)


def test_occlusion_interrupted_sequence_is_not_performed(verifier):
    face = synthetic_face()
    ok_hand = hand_over_face(face, "right", 1.3)
    ch = OcclusionChallenge("right")
    for f in _frames(face, [ok_hand] * 4 + [None] + [ok_hand] * 4):
        ch.add_frame(f, verifier)
    assert not ch.is_performed(verifier)


def test_occlusion_wrong_side_is_not_performed(verifier):
    face = synthetic_face()
    ch = OcclusionChallenge("left")
    for f in _frames(face, [hand_over_face(face, "right", 1.3)] * 10):
        ch.add_frame(f, verifier)
    assert not ch.is_performed(verifier)


def test_occlusion_uses_last_face_when_face_is_lost(verifier):
    face = synthetic_face()
    hand = hand_over_face(face, "mouth", 1.3)
    ch = OcclusionChallenge("mouth")
    ch.add_frame(FrameData(t=0.0, face=face, hand=None, size=SIZE), verifier)
    for i in range(1, 7):  # Face Mesh втратив обличчя під рукою
        ch.add_frame(FrameData(t=i / 30, face=None, hand=hand, size=SIZE), verifier)
    assert ch.is_performed(verifier)


# ----- поворот голови, solvePnP за 6 точками -----

@pytest.mark.parametrize("yaw", [-45, -30, -10, 0, 10, 30, 45])
def test_head_pose_recovers_yaw(verifier, yaw):
    est = verifier.head_pose(synthetic_face(yaw=yaw), SIZE)
    assert est[0] == pytest.approx(yaw, abs=1.0)
    assert abs(est[1]) < 1.0 and abs(est[2]) < 1.0


def test_head_pose_pitch_and_roll(verifier):
    assert verifier.head_pose(synthetic_face(pitch=15), SIZE)[1] == pytest.approx(15, abs=1.0)
    assert verifier.head_pose(synthetic_face(roll=-10), SIZE)[2] == pytest.approx(-10, abs=1.0)


def test_head_pose_with_landmark_noise(verifier):
    rng = np.random.default_rng(1)
    for yaw in (-35, 35):
        face = synthetic_face(yaw=yaw, noise_px=1.0, rng=rng)
        assert verifier.head_pose(face, SIZE)[0] == pytest.approx(yaw, abs=4.0)


@pytest.mark.parametrize("direction,yaw,expected", [
    ("left", 35, True), ("left", 25, False), ("left", -35, False),
    ("right", -35, True), ("right", 35, False),
])
def test_head_turn_challenge(verifier, direction, yaw, expected):
    ch = HeadTurnChallenge(direction)
    for i in range(6):
        ch.add_frame(FrameData(t=i / 30, face=synthetic_face(yaw=yaw), size=SIZE), verifier)
    assert ch.is_performed(verifier) is expected


# ----- проведення рукою -----

def _swipe(verifier, offsets):
    face = synthetic_face()
    fw = float(np.linalg.norm(face[454] - face[234]))
    ch = HandSwipeChallenge(min_shift=0.5)
    for i, d in enumerate(offsets):
        hand = synthetic_hand((face[1, 0] + d * fw, face[1, 1]), 0.5 * fw, 0.8 * fw)
        ch.add_frame(FrameData(t=i / 30, face=face, hand=hand, size=SIZE), verifier)
    return ch


def test_hand_swipe_crossing_midline(verifier):
    assert _swipe(verifier, np.linspace(-0.6, 0.6, 15)).is_performed(verifier)
    assert _swipe(verifier, np.linspace(0.6, -0.6, 15)).is_performed(verifier)


def test_hand_swipe_one_side_only(verifier):
    assert not _swipe(verifier, np.linspace(0.3, 0.9, 15)).is_performed(verifier)


def test_hand_swipe_too_small(verifier):
    assert not _swipe(verifier, np.linspace(-0.15, 0.15, 15)).is_performed(verifier)


def test_hand_offset_sign(verifier):
    face = synthetic_face()
    fw = float(np.linalg.norm(face[454] - face[234]))
    hand = synthetic_hand((face[1, 0] + 0.5 * fw, face[1, 1]), 0.5 * fw, 0.8 * fw)
    assert verifier.hand_offset(face, hand) == pytest.approx(0.5, abs=0.02)  # ліва сторона користувача
