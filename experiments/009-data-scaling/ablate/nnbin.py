"""BulletOu (2a8e5ed) の checkpoint / SFNN HalfKA2 nn.bin の読み書き (numpy 実装)。

再現する BulletOu 側の処理 (examples/bulletou.rs):
- `fold_sfnn_halfka2_piece_factorized_l0w` (9862-9888): nn.bin 行 r = l0w[r] + l0w[131949 + r % 1629] (f32 加算)
- `sfnn_quantise_i16` : (f64(v) * 127).round() (0.5 は 0 から遠い側) → clamp(i16) → i16
- `write_sfnn_leb128_i16_chunk` / `push_sfnn_signed_leb128_i16` (9891-9925):
  "COMPRESSED_LEB128" + u32 LE バイト数 + signed LEB128 の列
- `write_cuda_cpp_sfnn_nn_bin` (9543-9757): header → l0b chunk → l0w chunk → layer stack × 9

FT 以外 (header と layer stack) は FT の重みに依存しないので、元の nn.bin からバイト単位でコピーする。
"""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_BASE = 131_949  # HalfKA2 の特徴量数 (81 × 1629)
N_VIRTUAL = 1_629  # w_factor の行数 (駒入力 P)
FT_SIZE = 1_024
QA = 127.0
LEB128_MAGIC = b"COMPRESSED_LEB128"
SFNN_NNUE_VERSION = 0x7AF32F16


# ---------------------------------------------------------------- state.bin


def state_records(path: Path) -> dict[str, tuple[int, int]]:
    """state.bin (weights.bin) のレコード一覧: id -> (値の先頭バイトオフセット, 個数)。

    形式 (bulletou_lib/src/value/yaneuraou_kppt.rs:109): id + b'\\n' + u64 LE 個数 + f32 LE × 個数 の連結。
    値本体は読まずにヘッダだけを辿る。
    """
    recs: dict[str, tuple[int, int]] = {}
    size = os.path.getsize(path)
    off = 0
    with open(path, "rb") as f:
        while off < size:
            f.seek(off)
            head = f.read(512)
            nl = head.index(b"\n")
            rid = head[:nl].decode()
            (n,) = struct.unpack_from("<Q", head, nl + 1)
            recs[rid] = (off + nl + 9, n)
            off += nl + 9 + 4 * n
    if off != size:
        raise ValueError(f"{path}: trailing bytes ({off} != {size})")
    return recs


def load_record(path: Path, recs: dict[str, tuple[int, int]], rid: str) -> np.ndarray:
    off, n = recs[rid]
    return np.fromfile(path, dtype="<f4", count=n, offset=off)


def load_ft(state_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(l0w [131949+1629, 1024] f32, l0b [1024] f32) を畳み込み前のまま返す。"""
    recs = state_records(state_path)
    l0w = load_record(state_path, recs, "nnue/weights/l0w")
    if l0w.size != (N_BASE + N_VIRTUAL) * FT_SIZE:
        raise ValueError(f"unexpected l0w size {l0w.size} (factorized HalfKA2 1024 expected)")
    l0b = load_record(state_path, recs, "nnue/weights/l0b")
    return l0w.reshape(N_BASE + N_VIRTUAL, FT_SIZE), l0b


def find_state(ckpt: Path) -> Path:
    for name in ("state.bin", "weights.bin"):
        if (ckpt / name).exists():
            return ckpt / name
    raise FileNotFoundError(f"no state.bin / weights.bin in {ckpt}")


# ---------------------------------------------------------------- quantize / fold


def fold(l0w: np.ndarray) -> np.ndarray:
    """nn.bin 行 r = w_specific[r] + w_factor[r % 1629] (f32)。 [131949, 1024]"""
    spec = l0w[:N_BASE]
    fac = l0w[N_BASE:]
    out = np.empty_like(spec)
    for kb in range(N_BASE // N_VIRTUAL):
        s = slice(kb * N_VIRTUAL, (kb + 1) * N_VIRTUAL)
        np.add(spec[s], fac, out=out[s])
    return out


def quantize_i16(v: np.ndarray) -> np.ndarray:
    """Rust の (f64::from(v) * 127.0).round().clamp(i16) と同一。

    f32 × 127 は f64 で厳密 (有効 24+7 bit)。|x| < 2^52 なので trunc(x ± 0.5) は round-half-away と一致する。
    """
    x = v.astype(np.float64) * QA
    x = np.trunc(x + np.copysign(0.5, x))
    return np.clip(x, -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------- signed LEB128


def leb128_encode(q: np.ndarray) -> bytes:
    """i16 配列 → signed LEB128 (push_sfnn_signed_leb128_i16 と同一, 1〜3 バイト/値)。"""
    v = q.astype(np.int32).ravel()
    n = 1 + ((v < -64) | (v > 63)).astype(np.int64) + ((v < -8192) | (v > 8191)).astype(np.int64)
    off = np.cumsum(n) - n
    out = np.empty(int(n.sum()), dtype=np.uint8)
    out[off] = ((v & 0x7F) | np.where(n > 1, 0x80, 0)).astype(np.uint8)
    m2 = n >= 2
    out[off[m2] + 1] = (((v[m2] >> 7) & 0x7F) | np.where(n[m2] > 2, 0x80, 0)).astype(np.uint8)
    m3 = n == 3
    out[off[m3] + 2] = ((v[m3] >> 14) & 0x7F).astype(np.uint8)
    return out.tobytes()


def leb128_decode(buf: bytes | np.ndarray, count: int) -> np.ndarray:
    """signed LEB128 → i16 配列 (独立実装の検証用)。"""
    b = (
        np.frombuffer(buf, dtype=np.uint8)
        if isinstance(buf, (bytes, bytearray, memoryview))
        else buf
    )
    ends = np.flatnonzero((b & 0x80) == 0)
    if ends.size != count or (ends.size and ends[-1] != b.size - 1):
        raise ValueError(f"LEB128: {ends.size} values decoded, {count} expected")
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    ln = ends - starts + 1
    if ln.max() > 3:
        raise ValueError("LEB128: value longer than 3 bytes")
    v = (b[starts] & 0x7F).astype(np.int32)
    m = ln >= 2
    v[m] |= (b[starts[m] + 1] & 0x7F).astype(np.int32) << 7
    m = ln >= 3
    v[m] |= (b[starts[m] + 2] & 0x7F).astype(np.int32) << 14
    neg = (b[ends] & 0x40) != 0
    v[neg] -= np.left_shift(1, 7 * ln[neg]).astype(np.int32)
    if v.min(initial=0) < -32768 or v.max(initial=0) > 32767:
        raise ValueError("LEB128: value out of i16 range")
    return v.astype(np.int16)


def leb128_chunk(q: np.ndarray) -> bytes:
    payload = leb128_encode(q)
    return LEB128_MAGIC + struct.pack("<I", len(payload)) + payload


# ---------------------------------------------------------------- nn.bin


@dataclass
class NnBin:
    raw: bytes
    header: bytes  # version, hash, desc 長, desc, FT hash
    l0b_chunk: bytes  # magic + size + payload
    l0w_chunk: bytes
    tail: bytes  # layer stack × 9 (FT に依存しない)

    def l0b(self) -> np.ndarray:
        return leb128_decode(self.l0b_chunk[len(LEB128_MAGIC) + 4 :], FT_SIZE)

    def l0w(self) -> np.ndarray:
        return leb128_decode(self.l0w_chunk[len(LEB128_MAGIC) + 4 :], N_BASE * FT_SIZE).reshape(
            N_BASE, FT_SIZE
        )


def parse_nnbin(path: Path) -> NnBin:
    raw = Path(path).read_bytes()
    version, _hash, dlen = struct.unpack_from("<III", raw, 0)
    if version != SFNN_NNUE_VERSION:
        raise ValueError(f"{path}: version 0x{version:08X} is not SFNN")
    desc = raw[12 : 12 + dlen].decode()
    if f"[{N_BASE}->{FT_SIZE}x2]" not in desc or "HalfKA2" not in desc:
        raise ValueError(f"{path}: unexpected architecture {desc!r}")
    pos = 12 + dlen + 4
    header = raw[:pos]
    chunks = []
    for _ in range(2):
        if raw[pos : pos + len(LEB128_MAGIC)] != LEB128_MAGIC:
            raise ValueError(f"{path}: LEB128 magic not found at {pos}")
        (size,) = struct.unpack_from("<I", raw, pos + len(LEB128_MAGIC))
        end = pos + len(LEB128_MAGIC) + 4 + size
        chunks.append(raw[pos:end])
        pos = end
    return NnBin(raw, header, chunks[0], chunks[1], raw[pos:])


def build_nnbin(src: NnBin, l0b: np.ndarray, folded_l0w: np.ndarray) -> bytes:
    """header と layer stack は src からコピーし、FT (l0b, l0w) を f32 から量子化・符号化して差し込む。"""
    return (
        src.header
        + leb128_chunk(quantize_i16(l0b))
        + leb128_chunk(quantize_i16(folded_l0w))
        + src.tail
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()
