"""Геометричні функції для перевірки викликів з перекриттям.

Опуклі оболонки будуються методом монотонного ланцюга, перетин двох опуклих
многокутників – алгоритмом Сазерленда – Ходжмана, площа – формулою Гаусса.
"""

from __future__ import annotations

import numpy as np


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Повертає вершини опуклої оболонки з додатною орієнтованою площею."""
    pts = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def signed_area(poly: np.ndarray) -> float:
    """Орієнтована площа многокутника (формула Гаусса)."""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(poly: np.ndarray) -> float:
    """Площа многокутника S."""
    return abs(signed_area(poly))


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """Центр мас многокутника."""
    a = signed_area(poly)
    if abs(a) < 1e-9:
        return np.asarray(poly, dtype=float).mean(axis=0)
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    c = x * yn - xn * y
    return np.array([np.sum((x + xn) * c), np.sum((y + yn) * c)]) / (6.0 * a)


def clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Перетин многокутника subject з опуклим многокутником clip.

    Обидва многокутники мають бути задані з додатною орієнтованою площею
    (так їх повертає convex_hull).
    """
    output = [np.asarray(p, dtype=float) for p in subject]
    n = len(clip)
    for i in range(n):
        if not output:
            break
        a, b = clip[i], clip[(i + 1) % n]
        edge = b - a

        def inside(p):
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= 0

        def intersect(p, q):
            d = q - p
            denom = edge[0] * d[1] - edge[1] * d[0]
            if abs(denom) < 1e-12:
                return q
            s = (edge[1] * (p[0] - a[0]) - edge[0] * (p[1] - a[1])) / denom
            return p + s * d

        inp, output = output, []
        prev = inp[-1]
        for cur in inp:
            if inside(cur):
                if not inside(prev):
                    output.append(intersect(prev, cur))
                output.append(cur)
            elif inside(prev):
                output.append(intersect(prev, cur))
            prev = cur
    return np.array(output) if output else np.zeros((0, 2))
