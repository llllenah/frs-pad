"""Модуль виявлення атак підміни обличчя на основі активних викликів (FRS Platform).

Основні класи відповідають діаграмі класів (рисунок 3.10 пояснювальної записки):
SessionManager, ChallengeGenerator, ChallengeVerifier, DriftFeatureExtractor,
AttackClassifier (ThresholdClassifier, LogisticClassifier), PadSettings.
"""

from .challenges import (Challenge, ChallengeGenerator, HandSwipeChallenge, HeadTurnChallenge,
                         OcclusionChallenge)
from .classifiers import AttackClassifier, LogisticClassifier, ThresholdClassifier, load_classifier
from .features import FEATURE_NAMES, DriftFeatureExtractor
from .session import SessionManager, VerificationSession
from .settings import CHALLENGE_TYPES, PadSettings
from .structures import FeatureVector, FrameData, SessionDecision, SessionState
from .verifier import ChallengeVerifier

__version__ = "0.1.0"

__all__ = [
    "AttackClassifier", "CHALLENGE_TYPES", "Challenge", "ChallengeGenerator", "ChallengeVerifier",
    "DriftFeatureExtractor", "FEATURE_NAMES", "FeatureVector", "FrameData", "HandSwipeChallenge",
    "HeadTurnChallenge", "LogisticClassifier", "OcclusionChallenge", "PadSettings",
    "SessionDecision", "SessionManager", "SessionState", "ThresholdClassifier",
    "VerificationSession", "load_classifier",
]
