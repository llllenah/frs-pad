"""Вікно демонстрації сеансу верифікації (спрощена екранна форма рисунка 4.7).

Ліворуч – відео з ключовими точками обличчя та кисті (віддзеркалене, як у
дзеркалі), праворуч – панель з інструкцією, таймером, індикатором виконання,
графіком s_ref(t) і s_enr(t), рішенням і показниками швидкодії. OpenCV не
малює кирилицю, тому текст панелі виводиться засобами Pillow.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]

GREEN, ORANGE, RED, GRAY, WHITE = (80, 200, 80), (0, 160, 255), (60, 60, 230), (150, 150, 150), (240, 240, 240)
COLORS = {"bona_fide": GREEN, "attack": RED, "failed": GRAY}


def _font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def fmt(x: float | None, digits: int = 2) -> str:
    """Число з десятковою комою, як у тексті записки."""
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:.{digits}f}".replace(".", ",")


class DemoView:
    """Малює кадр демонстрації."""

    def __init__(self, video_width: int = 860, panel_width: int = 420, min_height: int = 580):
        self.video_width = video_width
        self.panel_width = panel_width
        self.min_height = min_height
        self.fonts = {k: _font(s) for k, s in (("big", 26), ("mid", 19), ("small", 15))}

    # ----- відео -----

    @staticmethod
    def draw_landmarks(frame: np.ndarray, data, box=None) -> None:
        """Точки Face Mesh (кожна четверта), кисть і рамка обличчя на кадрі."""
        if box is not None:
            x, y, w, h = map(int, box.bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        if data.face is not None:
            for p in data.face[::4].astype(int):
                cv2.circle(frame, tuple(p), 1, (255, 220, 120), -1)
        if data.hand is not None:
            pts = data.hand.astype(int)
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                         (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                         (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)):
                cv2.line(frame, tuple(pts[a]), tuple(pts[b]), ORANGE, 2)

    # ----- панель -----

    def _graph(self, panel: np.ndarray, top: int, height: int, series: dict, spans: list) -> None:
        left, right = 16, self.panel_width - 12
        bottom = top + height
        cv2.rectangle(panel, (left, top), (right, bottom), (70, 70, 70), 1)
        t, s_ref, s_enr = series["t"], series["s_ref"], series["s_enr"]
        if len(t) < 2:
            return
        t_max = max(float(t[-1]), 5.0)
        t_min = max(0.0, t_max - 20.0)

        def xy(tt, ss):
            x = left + (tt - t_min) / (t_max - t_min) * (right - left)
            y = bottom - np.clip(ss, 0, 1) * height
            return np.stack([x, y], axis=1)

        for a, b, color in spans:  # інтервали викликів
            xa, xb = xy(np.array([a, b]), np.zeros(2))[:, 0]
            if xb > left:
                cv2.rectangle(panel, (int(max(xa, left)), top + 1), (int(xb), bottom - 1), color, -1)
        for level in (0.25, 0.5, 0.75):
            y = int(bottom - level * height)
            cv2.line(panel, (left, y), (right, y), (55, 55, 55), 1)
        for s, color in ((s_enr, ORANGE), (s_ref, GREEN)):
            ok = np.isfinite(s) & (t >= t_min)
            if ok.sum() >= 2:
                cv2.polylines(panel, [xy(t[ok], s[ok]).astype(np.int32)], False, color, 2)

    def render(self, frame: np.ndarray, info: dict) -> np.ndarray:
        """Збирає кадр вікна: відео + панель. info – словник з даними для панелі."""
        h, w = frame.shape[:2]
        vh = round(h * self.video_width / w)
        height = max(vh, self.min_height)
        video = np.full((height, self.video_width, 3), 32, np.uint8)
        video[:vh] = cv2.resize(cv2.flip(frame, 1), (self.video_width, vh))
        panel = np.full((height, self.panel_width, 3), 32, np.uint8)

        status = info.get("status") or {}
        decision = info.get("decision")
        graph_top, graph_h = 250, 150
        spans = [(a, b, (60, 60, 45)) for a, b in info.get("spans", [])]
        self._graph(panel, graph_top, graph_h, info.get("series", {"t": []}), spans)
        prog = status.get("progress", 0.0)
        cv2.rectangle(panel, (16, 140), (self.panel_width - 12, 156), (70, 70, 70), 1)
        cv2.rectangle(panel, (17, 141), (17 + int(prog * (self.panel_width - 30)), 155), GREEN, -1)

        img = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        f = self.fonts
        d.text((16, 10), "Сеанс верифікації з активними викликами", font=f["mid"], fill=WHITE)
        state_line = f"Стан: {info.get('state_text', '–')}"
        if status.get("count"):
            state_line += f"   виклик {min(status['index'] + 1, status['count'])} з {status['count']}"
        if status.get("attempt", 1) > 1:
            state_line += f"   спроба {status['attempt']}"
        d.text((16, 38), state_line, font=f["small"], fill=GRAY[::-1])
        self._wrap(d, status.get("instruction", "Натисніть S, щоб почати сеанс"), (16, 62),
                   f["big"], WHITE, max_lines=2)
        if status.get("remaining") is not None:
            d.text((16, 160), f"Залишилось: {fmt(status['remaining'], 1)} с", font=f["mid"], fill=WHITE)
        y = 190
        for line in info.get("challenges", [])[:3]:
            d.text((16, y), line, font=f["small"], fill=WHITE)
            y += 18
        d.text((16, graph_top + graph_h + 4), "s_ref(t)", font=f["small"], fill=GREEN[::-1])
        d.text((90, graph_top + graph_h + 4), "s_enr(t)", font=f["small"], fill=ORANGE[::-1])
        d.text((170, graph_top + graph_h + 4), "останні 20 с, шкала 0–1", font=f["small"], fill=GRAY[::-1])
        y = graph_top + graph_h + 30
        if decision is not None:
            color = COLORS.get(decision["decision"], WHITE)[::-1]
            text = {"bona_fide": "Справжнє подання", "attack": "АТАКА",
                    "failed": "Не пройдено"}[decision["decision"]]
            if decision.get("score") is not None:
                text += f"  (оцінка {fmt(decision['score'])}, поріг {fmt(decision['threshold'])})"
            d.text((16, y), text, font=f["mid"], fill=color)
            y += 28
        for line in info.get("metrics", []):
            d.text((16, y), line, font=f["small"], fill=GRAY[::-1])
            y += 18
        d.text((16, height - 24), "S – сеанс   E – реєстрація обличчя   Q – вихід",
               font=f["small"], fill=GRAY[::-1])
        panel = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        return np.hstack([video, panel])

    def _wrap(self, d, text: str, xy, font, fill, max_lines: int = 2) -> None:
        words, lines, cur = text.split(), [], ""
        width = self.panel_width - 30
        for word in words:
            trial = (cur + " " + word).strip()
            if d.textlength(trial, font=font) <= width:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
        x, y = xy
        for line in lines[:max_lines]:
            d.text((x, y), line, font=font, fill=fill)
            y += font.size + 6 if hasattr(font, "size") else 28
