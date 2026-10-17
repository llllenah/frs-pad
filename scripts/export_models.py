"""Експорт попередньо навчених MTCNN і MobileFaceNet в ONNX та INT8-квантування MobileFaceNet.

Джерела ваг (моделі не донавчаються):
  * MTCNN (P-Net, R-Net, O-Net) – ваги з репозиторію timesler/facenet-pytorch
    (ліцензія MIT), перенесені з оригінальної реалізації Zhang et al. [10];
  * MobileFaceNet – ваги foamliu/MobileFaceNet v1.0 (ліцензія Apache 2.0),
    навчання на очищеному MS-Celeb-1M, ембединг 128, LFW 99,48 % за даними автора.

Потребує PyTorch (requirements-export.txt), тому запускається окремо від прототипу:

    python scripts/export_models.py --calib data/lfw/lfw

Результат у models/: mtcnn_pnet.onnx, mtcnn_rnet.onnx, mtcnn_onet.onnx,
mobilefacenet.onnx (FP32) і mobilefacenet_int8.onnx (статичне INT8-квантування
QDQ з калібруванням на вирівняних обличчях).
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCES = {
    "pnet.pt": ("https://raw.githubusercontent.com/timesler/facenet-pytorch/master/data/pnet.pt", None),
    "rnet.pt": ("https://raw.githubusercontent.com/timesler/facenet-pytorch/master/data/rnet.pt", None),
    "onet.pt": ("https://raw.githubusercontent.com/timesler/facenet-pytorch/master/data/onet.pt", None),
    "mobilefacenet.pt": ("https://github.com/foamliu/MobileFaceNet/releases/download/v1.0/mobilefacenet.pt", None),
}
SHA256 = {
    "pnet.pt": "a2a71925e0b9996a42f63e47efc1ca19043e69558b5c523b978d611dfae49c8f",
    "rnet.pt": "bbb937de72efc9ef83b186c49f5f558467a1d7e3453a8ece0d71a886633f6a86",
    "onet.pt": "165bfbe42940416ccfb977545cf0e976d5bf321f67083ae2aaaa5c764280118d",
    "mobilefacenet.pt": "90a00ba1d8b0b688af3deb731ed53dca582e6106805d1bc3cfdef55f570493f4",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, cache: Path) -> Path:
    dst = cache / name
    if not dst.exists():
        print(f"завантаження {name} ...")
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(SOURCES[name][0], tmp)
        tmp.replace(dst)
    digest = sha256(dst)
    if SHA256[name] and digest != SHA256[name]:
        raise SystemExit(f"хеш {name} не збігається: {digest}")
    print(f"{name}: sha256 {digest}")
    return dst


# --------------------------------------------------------------------------- MTCNN
# Архітектура мереж – як у facenet-pytorch (MIT), щоб ваги завантажувались без змін.

class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, 3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1)

    def forward(self, x):
        x = self.pool1(self.prelu1(self.conv1(x)))
        x = self.prelu3(self.conv3(self.prelu2(self.conv2(x))))
        prob = torch.softmax(self.conv4_1(x), dim=1)
        return self.conv4_2(x), prob


class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 28, 3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, 3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, 2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.dense5_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool1(self.prelu1(self.conv1(x)))
        x = self.pool2(self.prelu2(self.conv2(x)))
        x = self.prelu3(self.conv3(x))
        x = x.permute(0, 3, 2, 1).contiguous().reshape(x.shape[0], -1)
        x = self.prelu4(self.dense4(x))
        return self.dense5_2(x), torch.softmax(self.dense5_1(x), dim=1)


class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, 2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.prelu1(self.conv1(x)))
        x = self.pool2(self.prelu2(self.conv2(x)))
        x = self.pool3(self.prelu3(self.conv3(x)))
        x = self.prelu4(self.conv4(x))
        x = x.permute(0, 3, 2, 1).contiguous().reshape(x.shape[0], -1)
        x = self.prelu5(self.dense5(x))
        return self.dense6_2(x), self.dense6_3(x), torch.softmax(self.dense6_1(x), dim=1)


# --------------------------------------------------------------------------- MobileFaceNet
# Архітектура – як у foamliu/MobileFaceNet (Apache 2.0).

class ConvBNReLU(nn.Sequential):
    def __init__(self, inp, out, kernel_size=3, stride=1, groups=1):
        super().__init__(nn.Conv2d(inp, out, kernel_size, stride, (kernel_size - 1) // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(out), nn.ReLU6(inplace=True))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp, out, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(inp, inp, kernel_size, padding=padding, groups=inp, bias=False)
        self.pointwise = nn.Conv2d(inp, out, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.bn2 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        return self.relu(self.bn2(self.pointwise(x)))


class GDConv(nn.Module):
    def __init__(self, inp, out, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(inp, out, kernel_size, padding=padding, groups=inp, bias=False)
        self.bn = nn.BatchNorm2d(inp)

    def forward(self, x):
        return self.bn(self.depthwise(x))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        layers = [ConvBNReLU(inp, hidden, kernel_size=1)] if expand_ratio != 1 else []
        layers += [ConvBNReLU(hidden, hidden, stride=stride, groups=hidden),
                   nn.Conv2d(hidden, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class MobileFaceNet(nn.Module):
    SETTING = [[2, 64, 5, 2], [4, 128, 1, 2], [2, 128, 6, 1], [4, 128, 1, 2], [2, 128, 2, 1]]

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 64, stride=2)
        self.dw_conv = DepthwiseSeparableConv(64, 64, 3, 1)
        features, inp = [], 64
        for t, c, n, s in self.SETTING:
            for i in range(n):
                features.append(InvertedResidual(inp, c, s if i == 0 else 1, t))
                inp = c
        self.features = nn.Sequential(*features)
        self.conv2 = ConvBNReLU(inp, 512, kernel_size=1)
        self.gdconv = GDConv(512, 512, 7, 0)
        self.conv3 = nn.Conv2d(512, 128, 1)
        self.bn = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.gdconv(self.conv2(self.features(self.dw_conv(self.conv1(x)))))
        x = self.bn(self.conv3(x))
        return x.reshape(x.shape[0], -1)


# --------------------------------------------------------------------------- експорт
def export(model: nn.Module, dummy, path: Path, inputs, outputs, dynamic):
    model.eval()
    torch.onnx.export(model, dummy, str(path), input_names=inputs, output_names=outputs,
                      dynamic_axes=dynamic, opset_version=17, dynamo=False)
    print(f"{path.name}: {path.stat().st_size / 1e6:.2f} МБ")


def calibration_faces(lfw_dir: Path | None, n: int) -> list[np.ndarray]:
    """Вирівняні обличчя 112 × 112 (BGR) з LFW для калібрування INT8."""
    import cv2
    from pad.embedder import align_face
    from pad.mtcnn import MTCNN

    if lfw_dir is None or not lfw_dir.exists():
        raise SystemExit("для калібрування INT8 потрібен каталог LFW (--calib), див. scripts/eval_lfw.py --download")
    files = sorted(lfw_dir.rglob("*.jpg"))
    random.Random(0).shuffle(files)
    det = MTCNN(ROOT / "models")
    faces = []
    for f in files:
        img = cv2.imread(str(f))
        boxes = det.detect(img)
        if boxes:
            b = max(boxes, key=lambda x: x.score)
            faces.append(align_face(img, b.landmarks5))
        if len(faces) >= n:
            break
    return faces


def quantize_int8(fp32: Path, out: Path, faces: list[np.ndarray]) -> None:
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from pad.embedder import PROFILES, preprocess

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"input": preprocess(f, PROFILES["mobilefacenet"], 112)} for f in faces])

        def get_next(self):
            return next(self.it, None)

    pre = fp32.with_name(fp32.stem + "_pre.onnx")
    quant_pre_process(str(fp32), str(pre))
    quantize_static(str(pre), str(out), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    pre.unlink()
    print(f"{out.name}: {out.stat().st_size / 1e6:.2f} МБ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--calib", default=str(ROOT / "data" / "lfw" / "lfw"),
                    help="каталог із зображеннями облич для калібрування INT8 (LFW)")
    ap.add_argument("--calib-count", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out)
    cache = out / "src"
    cache.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    for name, cls, size, outs in (("pnet", PNet, 12, ["reg", "prob"]),
                                  ("rnet", RNet, 24, ["reg", "prob"]),
                                  ("onet", ONet, 48, ["reg", "points", "prob"])):
        net = cls()
        net.load_state_dict(torch.load(fetch(f"{name}.pt", cache), map_location="cpu"))
        dyn = {"input": {0: "n", 2: "h", 3: "w"}} if name == "pnet" else {"input": {0: "n"}}
        for o in outs:
            dyn[o] = {0: "n", 2: "h", 3: "w"} if name == "pnet" else {0: "n"}
        export(net, torch.zeros(1, 3, size, size), out / f"mtcnn_{name}.onnx", ["input"], outs, dyn)

    mfn = MobileFaceNet()
    mfn.load_state_dict(torch.load(fetch("mobilefacenet.pt", cache), map_location="cpu"))
    fp32 = out / "mobilefacenet.onnx"
    export(mfn, torch.zeros(1, 3, 112, 112), fp32, ["input"], ["embedding"],
           {"input": {0: "n"}, "embedding": {0: "n"}})
    quantize_int8(fp32, out / "mobilefacenet_int8.onnx", calibration_faces(Path(args.calib), args.calib_count))
    return 0


if __name__ == "__main__":
    sys.exit(main())
