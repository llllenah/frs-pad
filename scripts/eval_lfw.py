"""Точність верифікації на LFW (6000 пар, 10-кратна перехресна перевірка) і швидкодія моделей на CPU.

Протокол LFW «unrestricted with labeled outside data»: для кожної пари обличчя
знаходить MTCNN, вирівнювання – за 5 точками до 112 × 112, подібність –
косинус між L2-нормованими ембедингами. Поріг підбирається на 9 частинах і
перевіряється на десятій; наводиться середня точність і стандартне відхилення.
Якщо MTCNN не знаходить обличчя, береться центральна область 112 × 112 зображення
250 × 250 (як прийнято в оцінюванні на LFW).

Окремо вимірюються час детекції MTCNN на кадрі 1280 × 720, час ембединга
і розмір моделей. Результати зберігаються в runs/lfw_<модель>.json.

Запуск:
    python scripts/eval_lfw.py --download             # завантажити LFW і pairs.txt (≈ 180 МБ)
    python scripts/eval_lfw.py --embedder models/mobilefacenet_int8.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pad.cli import setup_console  # noqa: E402
from pad.embedder import OnnxFaceEmbedder, align_face  # noqa: E402
from pad.mtcnn import MTCNN  # noqa: E402

# дзеркало LFW на figshare (те саме, що використовує scikit-learn)
LFW_URL = "https://ndownloader.figshare.com/files/5976018"
PAIRS_URL = "https://ndownloader.figshare.com/files/5976006"


def download(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for url, name in ((LFW_URL, "lfw.tgz"), (PAIRS_URL, "pairs.txt")):
        if not (dst / name).exists():
            print(f"завантаження {name} ...")
            urllib.request.urlretrieve(url, dst / name)
    if not (dst / "lfw").exists():
        with tarfile.open(dst / "lfw.tgz") as tar:
            tar.extractall(dst)


def read_pairs(path: Path) -> list[tuple[str, str, bool]]:
    lines = path.read_text().strip().splitlines()[1:]
    pairs = []
    for line in lines:
        p = line.split()
        if len(p) == 3:
            pairs.append((f"{p[0]}/{p[0]}_{int(p[1]):04d}.jpg", f"{p[0]}/{p[0]}_{int(p[2]):04d}.jpg", True))
        else:
            pairs.append((f"{p[0]}/{p[0]}_{int(p[1]):04d}.jpg", f"{p[2]}/{p[2]}_{int(p[3]):04d}.jpg", False))
    return pairs


def best_threshold(sims: np.ndarray, labels: np.ndarray) -> float:
    cands = np.unique(sims)
    accs = [((sims >= t) == labels).mean() for t in cands]
    return float(cands[int(np.argmax(accs))])


def kfold_accuracy(sims: np.ndarray, labels: np.ndarray, folds: int = 10) -> tuple[float, float, float]:
    idx = np.arange(len(sims))
    parts = np.array_split(idx, folds)  # pairs.txt уже впорядковано за 10 частинами по 600 пар
    accs, ths = [], []
    for k in range(folds):
        test = parts[k]
        train = np.setdiff1d(idx, test)
        t = best_threshold(sims[train], labels[train])
        accs.append(((sims[test] >= t) == labels[test]).mean())
        ths.append(t)
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(ths))


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lfw", default=str(ROOT / "data" / "lfw"))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--detector", default=str(ROOT / "models"))
    ap.add_argument("--embedder", default=str(ROOT / "models" / "mobilefacenet_int8.onnx"))
    ap.add_argument("--profile", default="mobilefacenet")
    ap.add_argument("--flip", action="store_true", help="сума ембедингів зображення і його дзеркальної копії")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    lfw = Path(args.lfw)
    if args.download:
        download(lfw)
    pairs = read_pairs(lfw / "pairs.txt")
    det = MTCNN(args.detector, min_face=20, max_side=640)
    emb = OnnxFaceEmbedder(args.embedder, args.profile, threads=args.threads)

    cache: dict[str, np.ndarray] = {}
    misses = 0

    def embedding(rel: str) -> np.ndarray:
        nonlocal misses
        if rel in cache:
            return cache[rel]
        img = cv2.imread(str(lfw / "lfw" / rel))
        faces = det.detect(img)
        if faces:
            # обличчя, ближче до центру кадру (на LFW шукана особа – у центрі)
            c = np.array(img.shape[1::-1]) / 2
            f = min(faces, key=lambda b: np.linalg.norm(b.bbox[:2] + b.bbox[2:] / 2 - c))
            crop = align_face(img, f.landmarks5)
        else:
            misses += 1
            h, w = img.shape[:2]
            crop = img[h // 2 - 56:h // 2 + 56, w // 2 - 56:w // 2 + 56]
        e = emb.embed(crop)
        if args.flip:
            e = e + emb.embed(crop[:, ::-1])
            e = e / np.linalg.norm(e)
        cache[rel] = e
        return e

    t0 = time.perf_counter()
    sims, labels = [], []
    for i, (a, b, same) in enumerate(pairs):
        sims.append(float(embedding(a) @ embedding(b)))
        labels.append(same)
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(pairs)} пар")
    sims, labels = np.array(sims), np.array(labels)
    acc, std, th = kfold_accuracy(sims, labels)

    # швидкодія на CPU
    sample = cv2.imread(str(lfw / "lfw" / pairs[0][0]))
    frame = np.full((720, 1280, 3), 90, np.uint8)
    face = cv2.resize(sample, (500, 500))
    frame[110:610, 390:890] = face
    for _ in range(3):
        det.detect(frame)
    t = time.perf_counter()
    n = 30
    for _ in range(n):
        det.detect(frame)
    det_ms = (time.perf_counter() - t) / n * 1000
    crop = align_face(frame, det.detect(frame)[0].landmarks5)
    for _ in range(5):
        emb.embed(crop)
    t = time.perf_counter()
    for _ in range(100):
        emb.embed(crop)
    emb_ms = (time.perf_counter() - t) / 100 * 1000

    result = {
        "embedder": Path(args.embedder).name,
        "embedder_size_mb": round(Path(args.embedder).stat().st_size / 1e6, 2),
        "detector": "MTCNN (P/R/O-Net ONNX)",
        "detector_size_mb": round(sum((Path(args.detector) / f"mtcnn_{n}.onnx").stat().st_size
                                      for n in ("pnet", "rnet", "onet")) / 1e6, 2),
        "flip": args.flip,
        "pairs": len(pairs),
        "accuracy": round(acc, 4),
        "accuracy_std": round(std, 4),
        "threshold_mean": round(th, 4),
        "detection_misses": misses,
        "images": len(cache),
        "detect_ms_1280x720": round(det_ms, 1),
        "embed_ms": round(emb_ms, 2),
        "eval_time_s": round(time.perf_counter() - t0, 1),
    }
    out = ROOT / "runs" / f"lfw_{Path(args.embedder).stem}{'_flip' if args.flip else ''}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"збережено {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
