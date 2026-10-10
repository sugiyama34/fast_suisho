"""experiment-011: 出力側の経路を 1/k に縮めた nn.bin を書く (int8 の範囲に収めるため)。

PC-ALM で学習したネットは L3 の実効重み (stack の重み + shared factorizer) が int8 の範囲 (|w| ≤ 127/64) を超え、
BulletOu の書き出しで切り詰められて量子化後の評価が壊れる (notes/m0.md)。出力に効くのは
``out = W3eff · z2 + b3eff + z1[7]`` だけなので、L3 (重みと bias) と L1 の shortcut 行 (7 行目の重みと bias) を
同じ k で割ると、ネットの関数は出力が 1/k 倍になるだけ (FV_SCALE を 1/k にすれば探索での評価値は同じ)。

- 元の nn.bin (BulletOu が書いたもの) をそのまま読み、各 layer stack の L1 7 行目・L1 bias[7]・L3 重み・L3 bias だけを
  state.bin の float 値から量子化し直して書き換える (それ以外のバイトは変えない)
- k は ``--k`` か自動 (L3 と L1 7 行目の実効重みの最大 |w| が 127/64 に収まる最小の 2 のべき)。k = 1 なら元と同じバイト列
- 量子化は BulletOu と同じ: 重みは ×64 を四捨五入して int8 に clamp、bias は ×(127·64) を四捨五入して int32
- 出力: ``<out>/nn.bin`` と ``<out>/nn.bin.json`` (k, 元の nn.bin と state.bin の sha256, 範囲外だった割合)

使い方:
    .venv/bin/python experiments/011-pcalm/export_rescaled.py \
        --ckpt data/bulletou/checkpoints/011-s-pcalm-T2-gn/0001 --out /mnt/D/sugiyama/011/rescaled/011-s-pcalm-T2-gn-e1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "009-data-scaling" / "ablate"))
from nnbin import load_record, state_records  # noqa: E402

FT = 1024
L1_OUT = 8
L1_H = 7
L2 = 64
L2_IN_PAD = 32
STACKS = 9
QA = 127.0
QB = 64.0
W_MAX = 127.0 / QB
MAGIC = b"COMPRESSED_LEB128"
STACK_BYTES = 4 + 4 * L1_OUT + L1_OUT * FT + 4 * L2 + L2 * L2_IN_PAD + 4 + L2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def stack_offset(raw: bytes) -> int:
    """header と FT の 2 つの LEB128 チャンクを飛ばした位置 (最初の layer stack の先頭)。"""
    p = 8
    (dlen,) = struct.unpack_from("<I", raw, p)
    p += 4 + dlen + 4  # 説明文字列 + FT hash
    for _ in range(2):  # FT bias, FT weight
        assert raw[p : p + len(MAGIC)] == MAGIC, "LEB128 chunk expected"
        p += len(MAGIC)
        (n,) = struct.unpack_from("<I", raw, p)
        p += 4 + n
    assert len(raw) - p == STACKS * STACK_BYTES, (
        f"unexpected size: {len(raw) - p} != {STACKS * STACK_BYTES}"
    )
    return p


def q_i8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x.astype(np.float64) * QB), -128, 127).astype(np.int8)


def q_i32(x: np.ndarray) -> np.ndarray:
    return np.round(x.astype(np.float64) * QA * QB).astype(np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ckpt", required=True, type=Path, help="checkpoint (nn.bin と state.bin を含む)"
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--k", type=float, default=None, help="出力側の縮小率 (省略時は自動)")
    args = ap.parse_args()

    nn_path, st_path = args.ckpt / "nn.bin", args.ckpt / "state.bin"
    recs = state_records(st_path)

    def rec(name: str, shape: tuple[int, ...]) -> np.ndarray:
        return load_record(st_path, recs, f"nnue/weights/{name}").astype(np.float64).reshape(shape)

    l1w, l1b = rec("l1w", (STACKS, L1_OUT, FT)), rec("l1b", (STACKS, L1_OUT))
    l1fw, l1fb = rec("l1fw", (FT, L1_OUT)), rec("l1fb", (L1_OUT,))
    l3w, l3b = rec("l3w", (STACKS, L2)), rec("l3b", (STACKS,))
    l3fw, l3fb = rec("l3fw", (L2,)), rec("l3fb", (1,))
    w1_row7 = l1w[:, L1_H, :] + l1fw[:, L1_H][None, :]
    b1_7 = l1b[:, L1_H] + l1fb[L1_H]
    w1_rest = l1w[:, :L1_H, :] + l1fw[:, :L1_H].T[None, :, :]  # 縮められない行 (z1² の枝に入る)
    l2w, l2fw = rec("l2w", (STACKS, L2, 2 * L1_H)), rec("l2fw", (L2, 2 * L1_H))
    w2 = l2w + l2fw[None, :, :]
    w3 = l3w + l3fw[None, :]
    b3 = l3b + l3fb[0]
    max_abs = float(max(np.abs(w3).max(), np.abs(w1_row7).max()))
    k = (
        args.k
        if args.k is not None
        else (1.0 if max_abs <= W_MAX else 2.0 ** math.ceil(math.log2(max_abs / W_MAX)))
    )

    raw = bytearray(nn_path.read_bytes())
    base = stack_offset(bytes(raw))
    for s in range(STACKS):
        o = base + s * STACK_BYTES + 4  # stack hash の後
        struct.pack_into("<i", raw, o + 4 * L1_H, int(q_i32(np.array([b1_7[s] / k]))[0]))
        o_l1w = o + 4 * L1_OUT
        raw[o_l1w + L1_H * FT : o_l1w + (L1_H + 1) * FT] = q_i8(w1_row7[s] / k).tobytes()
        o_l3b = o_l1w + L1_OUT * FT + 4 * L2 + L2 * L2_IN_PAD
        struct.pack_into("<i", raw, o_l3b, int(q_i32(np.array([b3[s] / k]))[0]))
        raw[o_l3b + 4 : o_l3b + 4 + L2] = q_i8(w3[s] / k).tobytes()
    args.out.mkdir(parents=True, exist_ok=True)
    out_nn = args.out / "nn.bin"
    out_nn.write_bytes(bytes(raw))
    meta = {
        "k": k,
        "source_nn_bin": str(nn_path),
        "source_nn_bin_sha256": sha256(nn_path),
        "source_state_bin": str(st_path),
        "nn_bin_sha256": sha256(out_nn),
        "max_abs_w3eff": float(np.abs(w3).max()),
        "max_abs_w1eff_row7": float(np.abs(w1_row7).max()),
        "frac_w3eff_beyond_int8_before": float((np.abs(w3) > W_MAX).mean()),
        "frac_w3eff_beyond_int8_after": float((np.abs(w3 / k) > W_MAX).mean()),
        # 以下は縮めない層 (BulletOu の書き出しの int8 clip がそのまま効く)。0 でなければ量子化で関数が変わっている
        "max_abs_w1eff_rows0_6": float(np.abs(w1_rest).max()),
        "frac_w1eff_rows0_6_beyond_int8": float((np.abs(w1_rest) > W_MAX).mean()),
        "max_abs_w2eff": float(np.abs(w2).max()),
        "frac_w2eff_beyond_int8": float((np.abs(w2) > W_MAX).mean()),
        "note": "出力側 (L3 と L1 の shortcut 行) を 1/k 倍。探索での評価値を元と同じにするには FV_SCALE を 1/k 倍にする",
    }
    (args.out / "nn.bin.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
