"""nnbin.py の量子化・LEB128 を、BulletOu の Rust 関数を 1 値ずつ逐語移植した参照実装と突き合わせる。

学習済みの小さいネットは |q| ≤ 63 (1 バイト符号) に収まり、多バイト符号・clamp・0.5 の丸めを
実データでは踏まないため、合成データで全域を確認する。
    data/matchenv/bin/python experiments/009-data-scaling/ablate/test_nnbin.py
"""

from __future__ import annotations

import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nnbin  # noqa: E402


def ref_quantise_i16(v: float) -> int:  # sfnn_quantise_i16: f64 round() は 0.5 を 0 から遠い側へ
    x = float(np.float32(v)) * 127.0
    # Decimal(float) は厳密, HALF_UP = 0 から遠い側
    r = int(Decimal(x).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    return int(min(max(r, -32768), 32767))


def ref_leb128(value: int) -> bytes:  # push_sfnn_signed_leb128_i16
    out = bytearray()
    v = value
    while True:
        byte = v & 0x7F
        v >>= 7
        sign = byte & 0x40 != 0
        done = (v == 0 and not sign) or (v == -1 and sign)
        out.append(byte if done else byte | 0x80)
        if done:
            return bytes(out)


def main() -> None:
    # 1) LEB128: i16 の全値
    allv = np.arange(-32768, 32768, dtype=np.int32).astype(np.int16)
    enc = nnbin.leb128_encode(allv)
    assert enc == b"".join(ref_leb128(int(v)) for v in allv), "LEB128 encode mismatch"
    assert np.array_equal(nnbin.leb128_decode(enc, allv.size), allv), "LEB128 decode mismatch"
    # 2) 量子化: ちょうど .5 になる値, clamp 域, ±0, 乱数
    # v = 奇数/2 なら v*127 = 奇数*63.5 はちょうど .5 になる
    halves = (np.arange(-601, 602, 2) / 2).astype(np.float32)
    rng = np.random.default_rng(1)
    vals = np.concatenate(
        [
            halves,
            np.float32([0.0, -0.0, 1.98, -1.98, 258.0, -258.0, 300.0, -300.0, 1e9, -1e9]),
            (rng.standard_normal(200_000) * 3).astype(np.float32),
        ]
    )
    q = nnbin.quantize_i16(vals)
    ref = np.array([ref_quantise_i16(float(v)) for v in vals], dtype=np.int16)
    assert np.array_equal(q, ref), "quantize mismatch"
    x = vals.astype(np.float64) * 127
    print(
        f"OK: LEB128 all 65536 i16 values; quantize {vals.size} values "
        f"(exact .5 ties: {int((np.abs(x) % 1 == 0.5).sum())}, clamped: {int((np.abs(x) > 32767).sum())})"
    )


if __name__ == "__main__":
    main()
