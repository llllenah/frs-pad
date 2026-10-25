"""Перевірка виконання викликів за ключовими точками MediaPipe (ChallengeVerifier).

Геометричні методи (coverage, head_pose, hand_offset) працюють з масивами
точок у пікселях і не залежать від MediaPipe, тому їх перевіряють модульні
тести на синтетичних точках. MediaPipe Hands і Face Mesh створюються лише під
час першого виклику detect(), тобто лише під час сеансу верифікації.

Сторони «ліва» і «права» задано з погляду користувача. Кадр за замовчуванням
не віддзеркалений, тому ліва половина обличчя користувача – праворуч у кадрі.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import clip_convex, convex_hull, polygon_area, polygon_centroid

# Індекси точок Face Mesh.
FOREHEAD, CHIN, NOSE_TIP = 10, 152, 1
CHEEK_IMG_LEFT, CHEEK_IMG_RIGHT = 234, 454
#: 6 точок для solvePnP: кінчик носа, підборіддя, зовнішні кутики очей, кутики рота.
POSE_IDX = (NOSE_TIP, CHIN, 33, 263, 61, 291)
#: Узагальнена 3D-модель тих самих точок у системі камери (x – праворуч у кадрі,
#: y – донизу, z – від камери) для обличчя, повернутого прямо до камери.
POSE_MODEL = np.array([
    [0.0, 0.0, 0.0],          # кінчик носа
    [0.0, 330.0, 65.0],       # підборіддя
    [-225.0, -170.0, 135.0],  # зовнішній кутик ока ліворуч у кадрі
    [225.0, -170.0, 135.0],   # зовнішній кутик ока праворуч у кадрі
    [-150.0, 150.0, 125.0],   # кутик рота ліворуч у кадрі
    [150.0, 150.0, 125.0],    # кутик рота праворуч у кадрі
])
#: Точки долоні Hands (зап'ясток і основи пальців) для центру кисті.
PALM_IDX = (0, 5, 9, 13, 17)


class ChallengeVerifier:
    """Обчислює частку перекриття, кути повороту голови та зміщення кисті."""

    def __init__(self, mirrored: bool = False, max_num_hands: int = 1,
                 min_detection_confidence: float = 0.5, side_margin: float = 0.05,
                 palm_search_stride: int = 2):
        self.mirrored = mirrored
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.side_margin = side_margin
        self.palm_search_stride = max(1, palm_search_stride)
        self.hands = None
        self.face_mesh = None
        self._frame_no = 0
        self._hand_seen = False

    # ----- MediaPipe -----

    def _ensure_models(self) -> None:
        if self.face_mesh is not None:
            return
        import mediapipe as mp  # імпорт тут, щоб тести геометрії не залежали від MediaPipe

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=self.max_num_hands,
            model_complexity=0, min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=0.5)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=0.5)

    def detect(self, frame_bgr: np.ndarray,
               hands: bool = True) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Повертає точки обличчя (468, 2) і кисті (21, 2) у пікселях або None.

        hands=False пропускає Hands: коли кисті в кадрі немає, Hands щокадрово
        шукає долоню і займає більшу частину часу модуля.
        """
        self._ensure_models()
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        face = hand = None
        res = self.face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            face = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
        # поки кисть не знайдено, долоня шукається через кадр (пошук – найдорожча частина Hands);
        # знайдену кисть Hands відстежує на кожному кадрі
        self._frame_no += 1
        search = self._hand_seen or self._frame_no % self.palm_search_stride == 0
        res = self.hands.process(rgb) if hands and search else None
        self._hand_seen = bool(res is not None and res.multi_hand_landmarks)
        if res is not None and res.multi_hand_landmarks:
            hands = [np.array([(p.x * w, p.y * h) for p in hl.landmark], dtype=np.float32)
                     for hl in res.multi_hand_landmarks]
            hand = self._closest_hand(hands, face)
        return face, hand

    @staticmethod
    def _closest_hand(hands: list[np.ndarray], face: np.ndarray | None) -> np.ndarray:
        if face is None or len(hands) == 1:
            return hands[0]
        center = face.mean(axis=0)
        return min(hands, key=lambda h: float(np.linalg.norm(h[list(PALM_IDX)].mean(0) - center)))

    def close(self) -> None:
        """Звільняє ресурси MediaPipe."""
        for model in (self.hands, self.face_mesh):
            if model is not None:
                model.close()
        self.hands = self.face_mesh = None

    # ----- геометрія обличчя -----

    def _axes(self, face: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Середня лінія обличчя: початок, напрям донизу, нормаль до лівої сторони користувача, ширина."""
        top, chin = face[FOREHEAD].astype(float), face[CHIN].astype(float)
        down = chin - top
        down /= max(np.linalg.norm(down), 1e-9)
        normal = np.array([down[1], -down[0]])  # праворуч у кадрі
        if self.mirrored:
            normal = -normal
        width = float(np.linalg.norm(face[CHEEK_IMG_RIGHT] - face[CHEEK_IMG_LEFT]))
        return top, down, normal, max(width, 1e-6)

    def coverage_info(self, face: np.ndarray | None, hand: np.ndarray | None) -> dict:
        """Частка перекриття c(t) за формулою (3.3) і сторона перекриття.

        side – «left», «right» або «center» з погляду користувача; lower – чи
        лежить центр перекритої частини нижче кінчика носа; half_left, half_right –
        перекрита частка лівої і правої половини обличчя; lower_part – перекрита частка
        нижньої частини обличчя (нижче кінчика носа).
        """
        info = {"coverage": 0.0, "side": None, "lower": False, "half_left": 0.0, "half_right": 0.0,
                "lower_part": 0.0}
        if face is None or hand is None:
            return info
        face_hull = convex_hull(face)
        hand_hull = convex_hull(hand)
        face_area = polygon_area(face_hull)
        if face_area < 1e-6 or len(hand_hull) < 3:
            return info
        inter = clip_convex(hand_hull, face_hull)
        inter_area = polygon_area(inter) if len(inter) >= 3 else 0.0
        info["coverage"] = inter_area / face_area
        if inter_area <= 0:
            return info
        top, down, normal, width = self._axes(face)
        c = polygon_centroid(inter)
        lateral = float(np.dot(c - top, normal)) - float(np.dot(face[NOSE_TIP] - top, normal))
        if lateral > self.side_margin * width:
            info["side"] = "left"
        elif lateral < -self.side_margin * width:
            info["side"] = "right"
        else:
            info["side"] = "center"
        info["lower"] = bool(np.dot(c - face[NOSE_TIP], down) > 0)
        # частка перекриття кожної половини обличчя (межа – лінія через кінчик носа вздовж
        # середньої лінії): для cover_left / cover_right поріг задається від площі половини,
        # бо долоня на одній половині перекриває лише 0,2–0,35 площі всього обличчя
        nose = face[NOSE_TIP].astype(float)
        far = 10.0 * width
        for side, sign in (("left", 1.0), ("right", -1.0)):
            plane = convex_hull(np.array([nose - far * down, nose + far * down,
                                          nose + far * down + sign * far * normal,
                                          nose - far * down + sign * far * normal]))
            half = clip_convex(face_hull, plane)
            half_area = polygon_area(half) if len(half) >= 3 else 0.0
            part = clip_convex(inter, plane) if len(inter) >= 3 else []
            part_area = polygon_area(part) if len(part) >= 3 else 0.0
            info[f"half_{side}"] = part_area / half_area if half_area > 1e-6 else 0.0
        # нижня частина обличчя (рот і підборіддя) – нижче лінії через кінчик носа
        across = np.array([down[1], -down[0]])
        plane = convex_hull(np.array([nose - far * across, nose + far * across,
                                      nose + far * across + far * down, nose - far * across + far * down]))
        low = clip_convex(face_hull, plane)
        low_area = polygon_area(low) if len(low) >= 3 else 0.0
        part = clip_convex(inter, plane) if len(inter) >= 3 else []
        part_area = polygon_area(part) if len(part) >= 3 else 0.0
        info["lower_part"] = part_area / low_area if low_area > 1e-6 else 0.0
        return info

    def coverage(self, face: np.ndarray | None, hand: np.ndarray | None) -> float:
        """Частка площі обличчя, перекрита кистю, формула (3.3)."""
        return self.coverage_info(face, hand)["coverage"]

    def head_pose(self, face: np.ndarray, frame_size: tuple[int, int]) -> tuple[float, float, float]:
        """Кути (yaw, pitch, roll) у градусах, оцінені cv2.solvePnP за 6 точками.

        yaw > 0 – голову повернуто ліворуч (з погляду користувача),
        pitch > 0 – голову піднято, roll > 0 – нахил за годинниковою стрілкою в кадрі.
        """
        w, h = frame_size
        image_pts = np.asarray(face, dtype=np.float64)[list(POSE_IDX)]
        focal = float(w)
        camera = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
        # SQPNP шукає глобальний мінімум; ITERATIVE без початкового наближення за шуму
        # 1–2 пікселі часто переходить до дзеркального розв'язку зі зміною знака yaw
        ok, rvec, _ = cv2.solvePnP(POSE_MODEL, image_pts, camera, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return float("nan"), float("nan"), float("nan")
        rot, _ = cv2.Rodrigues(rvec)
        forward = rot @ np.array([0.0, 0.0, -1.0])  # напрям погляду обличчя
        yaw = math.degrees(math.atan2(forward[0], -forward[2]))
        pitch = math.degrees(math.atan2(-forward[1], math.hypot(forward[0], forward[2])))
        right = rot @ np.array([1.0, 0.0, 0.0])
        roll = math.degrees(math.atan2(right[1], right[0]))
        if self.mirrored:
            yaw, roll = -yaw, -roll
        return yaw, pitch, roll

    def hand_offset(self, face: np.ndarray | None, hand: np.ndarray | None) -> float | None:
        """Зміщення центру кисті від середньої лінії обличчя в частках ширини обличчя.

        Додатне значення – кисть з лівого боку користувача, від'ємне – з правого.
        """
        if face is None or hand is None:
            return None
        top, _, normal, width = self._axes(face)
        palm = np.asarray(hand, dtype=float)[list(PALM_IDX)].mean(axis=0)
        mid = float(np.dot(face[NOSE_TIP] - top, normal))
        return (float(np.dot(palm - top, normal)) - mid) / width
