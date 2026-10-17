"""Спільні фікстури тестів."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pad.verifier import ChallengeVerifier  # noqa: E402


@pytest.fixture
def verifier():
    return ChallengeVerifier()


@pytest.fixture
def rng():
    return np.random.default_rng(0)
