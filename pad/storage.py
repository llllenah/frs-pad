"""Шифрування записів пілотного набору даних (FR-65): AES-256-GCM.

Ключ – 32 байти в base64 у змінній середовища PAD_DATASET_KEY. У коді,
конфігурації чи поруч із записами ключ не зберігається, без нього записи
прочитати неможливо.

Формат файлу <ім'я>.enc: сигнатура FRSPAD01 (8 байт), випадковий nonce
(12 байт), шифротекст із тегом автентифікації (16 байт у кінці). Associated
data – ім'я відкритого файлу (video.mp4, session.json, <псевдонім>.npy), тому
перейменований або підмінений файл не розшифрується.

Так само зберігаються ембединги учасників (біометричні шаблони):
data/enrolled/<псевдонім>.npy.enc – масив у форматі .npy, зашифрований цілком.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_ENV = "PAD_DATASET_KEY"
KEY_SIZE = 32
NONCE_SIZE = 12
MAGIC = b"FRSPAD01"
SUFFIX = ".enc"


class DatasetKeyError(RuntimeError):
    """Ключ шифрування не задано або він має неправильний формат."""


class DecryptionError(ValueError):
    """Файл не розшифровується: інший ключ, пошкоджений або перейменований файл."""


def generate_key() -> str:
    """Новий випадковий ключ у base64 (значення для PAD_DATASET_KEY)."""
    return base64.b64encode(AESGCM.generate_key(bit_length=256)).decode("ascii")


def load_key(env: str = KEY_ENV) -> bytes:
    """Читає ключ зі змінної середовища env і перевіряє довжину."""
    value = os.environ.get(env, "").strip()
    if not value:
        raise DatasetKeyError(f"змінну середовища {env} не задано (32 байти в base64)")
    try:
        key = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise DatasetKeyError(f"{env} не є рядком base64") from exc
    if len(key) != KEY_SIZE:
        raise DatasetKeyError(f"{env} має містити {KEY_SIZE} байти, а не {len(key)}")
    return key


def load_key_if_set(env: str = KEY_ENV) -> bytes | None:
    """Ключ зі змінної env або None, якщо її не задано (неправильний ключ – DatasetKeyError)."""
    return load_key(env) if os.environ.get(env, "").strip() else None


def plain_name(path: str | Path) -> str:
    """Ім'я відкритого файлу: session.json.enc -> session.json."""
    name = Path(path).name
    return name[:-len(SUFFIX)] if name.endswith(SUFFIX) else name


def encrypt_bytes(data: bytes, name: str, key: bytes) -> bytes:
    """Шифрує data; name (ім'я відкритого файлу) стає associated data."""
    nonce = os.urandom(NONCE_SIZE)
    return MAGIC + nonce + AESGCM(key).encrypt(nonce, data, name.encode("utf-8"))


def decrypt_bytes(blob: bytes, name: str, key: bytes) -> bytes:
    """Розшифровує результат encrypt_bytes з тим самим name."""
    head = len(MAGIC) + NONCE_SIZE
    if len(blob) < head + 16 or not blob.startswith(MAGIC):
        raise DecryptionError("це не зашифрований запис FRS PAD")
    nonce = blob[len(MAGIC):head]
    try:
        return AESGCM(key).decrypt(nonce, blob[head:], name.encode("utf-8"))
    except InvalidTag as exc:
        raise DecryptionError("неправильний ключ або файл пошкоджено чи перейменовано") from exc


def write_encrypted(path: str | Path, data: bytes, key: bytes | None = None) -> Path:
    """Записує data зашифрованими у path.enc, не створюючи відкритого файлу."""
    path = Path(path)
    out = path.with_name(path.name + SUFFIX)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(encrypt_bytes(data, path.name, key or load_key()))
    tmp.replace(out)
    return out


def encrypt_file(path: str | Path, key: bytes | None = None, keep_plain: bool = False) -> Path:
    """Шифрує файл path у path.enc і (за замовчуванням) видаляє відкритий файл."""
    path = Path(path)
    out = write_encrypted(path, path.read_bytes(), key)
    if not keep_plain:
        path.unlink()
    return out


def read_encrypted(path: str | Path, key: bytes | None = None) -> bytes:
    """Вміст зашифрованого файлу path (.enc) у пам'яті."""
    path = Path(path)
    try:
        return decrypt_bytes(path.read_bytes(), plain_name(path), key or load_key())
    except DecryptionError as exc:
        raise DecryptionError(f"{path}: {exc}") from exc


def read_encrypted_buffer(path: str | Path, key: bytes | None = None, chunk: int = 16 << 20) -> bytearray:
    """Вміст великого зашифрованого файлу (відео) у пам'яті, без проміжної копії шифротексту.

    Файл читається частинами по chunk байт і розшифровується потоково в заздалегідь
    виділений буфер; тег GCM перевіряється в кінці (finalize). До перевірки тегу
    буфер нікуди не передається, тому підмінене відео не потрапить до декодера.
    """
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    path = Path(path)
    key = key or load_key()
    head = len(MAGIC) + NONCE_SIZE
    size = path.stat().st_size
    if size < head + 16:
        raise DecryptionError(f"{path}: це не зашифрований запис FRS PAD")
    n = size - head - 16
    with open(path, "rb") as fh:
        if fh.read(len(MAGIC)) != MAGIC:
            raise DecryptionError(f"{path}: це не зашифрований запис FRS PAD")
        nonce = fh.read(NONCE_SIZE)
        fh.seek(size - 16)
        tag = fh.read(16)
        fh.seek(head)
        dec = Cipher(algorithms.AES(key), modes.GCM(nonce, tag)).decryptor()
        dec.authenticate_additional_data(plain_name(path).encode("utf-8"))
        buf = bytearray(n + 16)
        view = memoryview(buf)
        pos = 0
        try:
            while pos < n:
                part = fh.read(min(chunk, n - pos))
                if not part:
                    raise DecryptionError(f"{path}: файл обрізано")
                pos += dec.update_into(part, view[pos:pos + len(part) + 15])
            dec.finalize()
        except InvalidTag as exc:
            raise DecryptionError(f"{path}: неправильний ключ або файл пошкоджено чи перейменовано") from exc
        finally:
            view.release()
    del buf[pos:]
    return buf


def decrypt_file(path: str | Path, out: str | Path | None = None, key: bytes | None = None) -> Path:
    """Розшифровує path (.enc) у файл out (за замовчуванням – поруч, без .enc)."""
    path = Path(path)
    out = Path(out) if out is not None else path.with_name(plain_name(path))
    out.write_bytes(read_encrypted(path, key))
    return out


def read_json_encrypted(path: str | Path, key: bytes | None = None) -> dict:
    """Читає зашифрований JSON (session.json.enc)."""
    return json.loads(read_encrypted(path, key).decode("utf-8"))


def read_json(path: str | Path, key: bytes | None = None) -> dict:
    """Читає session.json або session.json.enc (ключ потрібен лише для .enc)."""
    path = Path(path)
    if path.name.endswith(SUFFIX):
        return read_json_encrypted(path, key)
    return json.loads(path.read_text(encoding="utf-8"))


def save_embedding(path: str | Path, emb: np.ndarray, key: bytes | None = None) -> Path:
    """Зберігає ембединг у path.enc (<псевдонім>.npy.enc), відкритий .npy не створюється.

    Масив серіалізується np.save у пам'ять. Відкритий path з попередньої реєстрації
    видаляється, бо його замінює новий зашифрований ембединг.
    """
    path = Path(path).with_name(plain_name(path))
    buf = io.BytesIO()
    np.save(buf, np.asarray(emb))
    out = write_encrypted(path, buf.getvalue(), key)
    path.unlink(missing_ok=True)
    return out


def load_embedding(path: str | Path, key: bytes | None = None) -> np.ndarray:
    """Читає ембединг з <псевдонім>.npy.enc або відкритого .npy (ключ потрібен лише для .enc)."""
    path = Path(path)
    if path.name.endswith(SUFFIX):
        return np.load(io.BytesIO(read_encrypted(path, key)))
    return np.load(path)


def find_embedding(path: str | Path) -> Path | None:
    """Наявний файл ембединга для path (<псевдонім>.npy): спершу .npy.enc, потім .npy."""
    plain = Path(path).with_name(plain_name(path))
    for candidate in (plain.with_name(plain.name + SUFFIX), plain):
        if candidate.exists():
            return candidate
    return None
