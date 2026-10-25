"""Генерування атак підміни обличчя з записаного відео (без GPU, у фоні).

Справжній запис учасника A (з remote/remote_recorder.html) обробляється
кадр за кадром моделлю inswapper_128 (як у Deep-Live-Cam): обличчя A
замінюється обличчям цільової особи B. Виклики, часові мітки і поведінка
лишаються з оригінального запису, тому анотація A підходить і для атаки.

    python scripts/generate_attacks.py A.webm --source-video B.webm --source-annotation B.json -o A_as_B.avi
    python scripts/import_remote.py A_as_B.avi A.json --type attack --tool deep_live_cam --target B

Справжній запис для набору спершу проходить той самий скрипт без підміни:

    python scripts/generate_attacks.py A.webm --passthrough -o A.avi
    python scripts/import_remote.py A.avi A.json

Так обидва класи мають однакові кадри, мітки часу і кодек (MJPG, якість 95),
і класифікатор не може відрізнити атаку за частотою кадрів чи артефактами
повторного стиснення. Кожен кадр входу дає один кадр виходу, мітки часу
оригіналу (webm має змінну частоту кадрів) записуються в <вихід>.times.json,
і import_remote.py бере їх звідти.

Якщо на кадрі обличчя не знайдено (наприклад, його закрито рукою), кадр
лишається без змін – так само поводиться Deep-Live-Cam, і саме цей ефект
перевіряють виклики з перекриттям.

Швидкість inswapper_128 на CPU – 0,3–2 кадри/с залежно від процесора
(на ноутбуці автора ≈ 2,7 с на кадр, тобто ≈ 1–1,5 год на 1 хв відео);
запуск можна лишити у фоні на ніч. У Google Colab з безкоштовним GPU T4
(notebooks/generate_attacks_colab.ipynb) – десятки кадрів/с.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pad.cli import setup_console  # noqa: E402
from pad.mtcnn import MTCNN  # noqa: E402
from pad.swapper import ArcFace, InSwapper, ensure_models  # noqa: E402

MIN_SCORE = 0.9


def largest_face(detector: MTCNN, frame: np.ndarray):
    faces = [f for f in detector.detect(frame) if f.score >= MIN_SCORE]
    if not faces:
        return None
    return max(faces, key=lambda f: f.bbox[2] * f.bbox[3])  # рамка (x, y, w, h)


def source_embedding(arcface: ArcFace, detector: MTCNN, image: Path | None = None,
                     video: Path | None = None, annotation: dict | None = None, max_frames: int = 30) -> np.ndarray:
    """Ембединг цільової особи: з фото або середнє по кадрах вікна реєстрації її відео."""
    frames = []
    if image is not None:
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"не вдалося прочитати {image}")
        frames.append(img)
    else:
        t0, t1 = 0.0, 1e9
        if annotation and annotation.get("enroll"):
            t0, t1 = annotation["enroll"]["t_start"], annotation["enroll"]["t_end"]
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(f"не вдалося відкрити відео {video}")
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t0 <= t <= t1:
                frames.append(frame)
            elif t > t1:
                break
        cap.release()
    embs = []
    for frame in frames:
        face = largest_face(detector, frame)
        if face is not None:
            embs.append(arcface.embed(frame, face.landmarks5))
    if not embs:
        raise ValueError("на джерелі цільової особи не знайдено обличчя")
    e = np.mean(embs, axis=0)
    return e / np.linalg.norm(e)


def times_path(video: Path) -> Path:
    """Файл з мітками часу кадрів, який супроводжує відео з generate_attacks.py."""
    return video.with_name(video.name + ".times.json")


def swap_video(video: Path, out: Path, latent: np.ndarray | None, swapper: InSwapper | None,
               detector: MTCNN | None, log=print) -> dict:
    """Підміняє обличчя в кожному кадрі; latent=None – копія без підміни.

    Кожен кадр входу дає рівно один кадр виходу, а мітки часу кадрів оригіналу
    зберігаються поруч (<вихід>.times.json) і використовуються при імпорті. Справжні
    записи для набору проходять ту саму функцію без підміни, тому обидва класи мають
    однакові кадри, мітки часу і кодек (MJPG, якість 95) – класифікатор не може
    відрізнити атаку за частотою кадрів чи артефактами стиснення.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"не вдалося відкрити відео {video}")
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if not fps_in or fps_in > 120:
        fps_in = 30.0
    frames, times = [], []
    prev_t = -1.0
    i = swapped = 0
    started = time.perf_counter()
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        t = pos if prev_t < pos < 1e5 else i / fps_in
        if t <= prev_t:
            t = prev_t + 1.0 / fps_in
        prev_t = t
        i += 1
        if latent is not None:
            face = largest_face(detector, frame)
            if face is not None:
                frame = swapper.swap(frame, face.landmarks5, latent)
                swapped += 1
        if writer is None:
            h, w = frame.shape[:2]
            out.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h))
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
        writer.write(frame)
        times.append(round(t, 4))
        if latent is not None and i % 150 == 0:
            el = time.perf_counter() - started
            log(f"  {t:.1f} с відео, кадр {i}, {i / el:.2f} кадр/с")
    cap.release()
    if writer is not None:
        writer.release()
        tmp.replace(out)
        times_path(out).write_text(json.dumps(times), encoding="utf-8")
    return {"frames": i, "swapped": swapped, "seconds": round(time.perf_counter() - started, 1)}


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="справжній запис учасника A (.webm або .mp4)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--passthrough", action="store_true",
                     help="без підміни: копія справжнього запису з тими самими кадрами і кодеком")
    src.add_argument("--source-image", help="фото цільової особи B")
    src.add_argument("--source-video", help="справжній запис цільової особи B")
    ap.add_argument("--source-annotation", help="анотація запису B (береться вікно enroll)")
    ap.add_argument("-o", "--out", required=True, help="вихідне відео .avi (MJPG)")
    ap.add_argument("--models", default=str(ROOT / "models"), help="каталог з mtcnn_*.onnx")
    ap.add_argument("--swap-models", default=str(ROOT / "models" / "swap"),
                    help="каталог inswapper_128.onnx і w600k_r50.onnx (завантажуються, якщо немає)")
    ap.add_argument("--cpu", action="store_true", help="не використовувати CUDA навіть за наявності")
    args = ap.parse_args()
    if args.passthrough:
        stats = swap_video(Path(args.video), Path(args.out), None, None, None)
        print(f"готово: {args.out}; кадрів {stats['frames']}, {stats['seconds']} с")
        return 0

    swap_dir = ensure_models(args.swap_models)
    providers = ["CPUExecutionProvider"] if args.cpu else None
    detector = MTCNN(args.models, min_face=40, max_side=640)
    arcface = ArcFace(swap_dir / "w600k_r50.onnx", providers)
    swapper = InSwapper(swap_dir / "inswapper_128.onnx", providers)
    print("провайдер:", swapper.session.get_providers()[0])

    ann = json.loads(Path(args.source_annotation).read_text(encoding="utf-8")) if args.source_annotation else None
    emb = source_embedding(arcface, detector,
                           image=Path(args.source_image) if args.source_image else None,
                           video=Path(args.source_video) if args.source_video else None, annotation=ann)
    stats = swap_video(Path(args.video), Path(args.out), swapper.latent(emb), swapper, detector)
    print(f"готово: {args.out}; кадрів {stats['frames']}, "
          f"обличчя підмінено на {stats['swapped']} кадрах, {stats['seconds']} с")
    if stats["frames"] and stats["swapped"] / stats["frames"] < 0.5:
        print("увага: обличчя підмінено менше ніж на половині кадрів, перевірте відео")
    return 0


if __name__ == "__main__":
    sys.exit(main())
