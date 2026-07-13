"""signed LEB128 の numpy ベクトル化コーデック。

やねうら王 `nnue_common.h` の `read_leb_128` と互換のフォーマットを扱う。
1 ブロック = magic "COMPRESSED_LEB128" + uint32 (圧縮バイト数) + 圧縮データ。
エンコードは最短長 (canonical) 表現で、nnue-pytorch / Stockfish の
`write_leb_128` と同一の出力を生成する (Suisho 11 nn.bin とビット一致することを
tests/tools/test_roundtrip.py で検証している)。
"""

from __future__ import annotations

import numpy as np

LEB128_MAGIC = b"COMPRESSED_LEB128"


def _max_encoded_len(dtype: np.dtype) -> int:
    # 7bit グループ数の上限。int16 なら 3 バイト (shift 0/7/14)
    return (dtype.itemsize * 8 + 6) // 7


def decode(data: bytes | np.ndarray, count: int, dtype: type = np.int16) -> np.ndarray:
    """圧縮バイト列 (magic/長さヘッダを除いた本体) を count 個の整数へ復号する。"""
    dtype = np.dtype(dtype)
    if dtype.kind != "i":
        raise ValueError("signed integer dtype required")
    bits = dtype.itemsize * 8
    max_len = _max_encoded_len(dtype)

    raw = np.frombuffer(data, dtype=np.uint8) if not isinstance(data, np.ndarray) else data
    ends = np.flatnonzero(raw < 0x80)
    if len(ends) != count:
        raise ValueError(f"expected {count} varints, found {len(ends)}")
    if count == 0:
        if len(raw) != 0:
            raise ValueError("trailing bytes after last varint")
        return np.empty(0, dtype=dtype)
    if ends[-1] != len(raw) - 1:
        raise ValueError("trailing bytes after last varint")
    starts = np.empty(count, dtype=np.int64)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    lens = ends - starts + 1
    if lens.max() > max_len:
        raise ValueError(f"varint longer than {max_len} bytes for {dtype}")

    acc = np.zeros(count, dtype=np.uint64)
    for k in range(max_len):
        mask = lens > k
        acc[mask] |= (raw[starts[mask] + k] & np.uint64(0x7F)).astype(np.uint64) << np.uint64(7 * k)

    # 最終バイトの bit6 が立っていて、かつ 7*len < bits のとき符号拡張する
    shift = (7 * lens).astype(np.uint64)
    ext = (shift < bits) & ((raw[ends] & 0x40) != 0)
    acc[ext] |= (~np.uint64(0)) << shift[ext]

    mask = np.uint64((1 << bits) - 1)
    unsigned_dtype = np.dtype(f"u{dtype.itemsize}")
    return (acc & mask).astype(unsigned_dtype).view(dtype)


def encode(values: np.ndarray) -> bytes:
    """整数配列を最短長 signed LEB128 バイト列 (本体のみ) へ符号化する。"""
    dtype = values.dtype
    if dtype.kind != "i":
        raise ValueError("signed integer dtype required")
    max_len = _max_encoded_len(dtype)

    v = values.reshape(-1).astype(np.int64)
    if v.size == 0:
        return b""
    # 最短長: 残り (v >> 7k) が 0 か -1 で、直前グループの bit6 が符号と一致した時点で終端
    lens = np.full(v.shape, max_len, dtype=np.int64)
    for k in range(max_len - 1, 0, -1):
        fits = (v >= -(1 << (7 * k - 1))) & (v < (1 << (7 * k - 1)))
        lens[fits] = k

    ends = np.cumsum(lens)
    starts = ends - lens
    out = np.empty(int(ends[-1]), dtype=np.uint8)
    for k in range(max_len):
        mask = lens > k
        byte = ((v[mask] >> (7 * k)) & 0x7F).astype(np.uint8)
        cont = (lens[mask] > k + 1).astype(np.uint8) << 7
        out[starts[mask] + k] = byte | cont
    return out.tobytes()


def read_block(
    buf: bytes, offset: int, count: int, dtype: type = np.int16
) -> tuple[np.ndarray, int]:
    """buf[offset:] から magic + 長さ + 本体のブロックを読み、(配列, 次 offset) を返す。"""
    magic = buf[offset : offset + len(LEB128_MAGIC)]
    if magic != LEB128_MAGIC:
        raise ValueError(f"LEB128 magic not found at offset {offset}")
    offset += len(LEB128_MAGIC)
    size_field = buf[offset : offset + 4]
    if len(size_field) != 4:
        raise ValueError(f"truncated LEB128 length field at offset {offset}")
    nbytes = int.from_bytes(size_field, "little")
    offset += 4
    if offset + nbytes > len(buf):
        raise ValueError(f"truncated LEB128 body at offset {offset}")
    body = np.frombuffer(buf, dtype=np.uint8, count=nbytes, offset=offset)
    return decode(body, count, dtype), offset + nbytes


def write_block(values: np.ndarray) -> bytes:
    """配列を magic + 長さ + 本体のブロックへ符号化する。"""
    body = encode(values)
    return LEB128_MAGIC + len(body).to_bytes(4, "little") + body
