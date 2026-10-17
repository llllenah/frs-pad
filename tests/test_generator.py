"""Тести генератора активних викликів (FR-51)."""

import random
import secrets

import pytest

from pad.challenges import (ChallengeGenerator, HandSwipeChallenge, HeadTurnChallenge,
                            OcclusionChallenge)
from pad.settings import CHALLENGE_TYPES, PadSettings


def test_uses_system_random_by_default():
    assert isinstance(ChallengeGenerator().rng, secrets.SystemRandom)


def test_at_least_four_types_available():
    assert len(CHALLENGE_TYPES) >= 4


def test_default_count_is_three():
    gen = ChallengeGenerator()
    assert len(gen.generate()) == 3
    assert PadSettings().challenge_count == 3


def test_no_two_identical_in_a_row():
    gen = ChallengeGenerator()
    for _ in range(2000):
        seq = gen.generate_types(5)
        assert all(a != b for a, b in zip(seq, seq[1:])), seq


def test_previous_type_is_not_repeated_on_retry():
    gen = ChallengeGenerator()
    for _ in range(500):
        assert gen.generate_types(3, previous="turn_head")[0] != "turn_head"


def test_all_types_are_used():
    gen = ChallengeGenerator(rng=random.Random(1))
    seen = {k for _ in range(300) for k in gen.generate_types(3)}
    assert seen == set(CHALLENGE_TYPES)


def test_respects_allowed_subset():
    gen = ChallengeGenerator()
    allowed = ["cover_left", "turn_head"]
    for _ in range(200):
        seq = gen.generate_types(4, allowed)
        assert set(seq) <= set(allowed)
        assert seq in (["cover_left", "turn_head"] * 2, ["turn_head", "cover_left"] * 2)


def test_sequences_are_not_predictable():
    gen = ChallengeGenerator()
    sequences = {tuple(gen.generate_types(3)) for _ in range(500)}
    assert len(sequences) > 30  # 5 · 4 · 4 = 80 можливих послідовностей


def test_object_types_and_instructions():
    gen = ChallengeGenerator(rng=random.Random(3))
    objs = [c for _ in range(100) for c in gen.generate(3)]
    kinds = {type(c) for c in objs}
    assert kinds == {OcclusionChallenge, HeadTurnChallenge, HandSwipeChallenge}
    for c in objs:
        assert c.instruction()
        assert c.duration_s <= 5.0
    directions = {c.direction for c in objs if isinstance(c, HeadTurnChallenge)}
    assert directions == {"left", "right"}


@pytest.mark.parametrize("n,types", [(0, None), (2, ["cover_left"]), (3, ["unknown"])])
def test_invalid_arguments(n, types):
    with pytest.raises(ValueError):
        ChallengeGenerator().generate_types(n, types)


def test_settings_validation():
    with pytest.raises(ValueError):
        PadSettings(challenge_types=["cover_left", "cover_right", "turn_head"]).validate()
    with pytest.raises(ValueError):
        PadSettings(challenge_count=6).validate()
    with pytest.raises(ValueError):
        PadSettings(duration_s=6.0).validate()
    with pytest.raises(ValueError):
        PadSettings(max_retries=3).validate()
    PadSettings().validate()
