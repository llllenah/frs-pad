"""Завантажує запасні ONNX-моделі з OpenCV Zoo (ліцензія Apache 2.0).

Основні моделі з ТЗ – MTCNN і MobileFaceNet INT8 – готує scripts/export_models.py
(потрібен PyTorch, див. README). YuNet (детектор облич, 5 ключових точок) і SFace
(128-вимірний ембединг) лишаються запасним варіантом для configs/pad.yaml.

За замовчуванням завантажується SFace FP32 (37 МБ): у ONNX Runtime на CPU вона
працює приблизно за 15 мс, тоді як квантована SFace INT8 – за 90–130 мс (її
оператори погано оптимізуються в ONNX Runtime), тому INT8 – лише за --int8.

Запуск: python scripts/download_models.py [--int8]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pad.cli import setup_console  # noqa: E402

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models/"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "face_recognition_sface_2021dec_int8.onnx": (
        "face_recognition_sface/face_recognition_sface_2021dec_int8.onnx",
        "2b0e941e6f16cc048c20aee0c8e31f569118f65d702914540f7bfdc14048d78a"),
    "face_recognition_sface_2021dec.onnx": (
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "models"))
    ap.add_argument("--int8", action="store_true", help="також завантажити SFace INT8 (9,9 МБ)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in MODELS if args.int8 or not n.endswith("_int8.onnx")]
    ok = True
    for name in names:
        rel, digest = MODELS[name]
        dst = out / name
        if not dst.exists():
            print(f"завантаження {name} ...")
            tmp = dst.with_suffix(".part")
            urllib.request.urlretrieve(BASE + rel, tmp)
            tmp.replace(dst)
        actual = sha256(dst)
        if digest and actual != digest:
            print(f"ПОМИЛКА: хеш {name} не збігається ({actual})")
            ok = False
        else:
            print(f"{name}: {dst.stat().st_size / 1e6:.1f} МБ, sha256 {actual[:16]}…")
    from pad.mtcnn import MTCNN

    if not (MTCNN.available(out) and (out / "mobilefacenet_int8.onnx").exists()):
        print("MTCNN і MobileFaceNet INT8 ще не підготовлено: запустіть scripts/export_models.py "
              "у середовищі з PyTorch (README, розділ «Встановлення»)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
