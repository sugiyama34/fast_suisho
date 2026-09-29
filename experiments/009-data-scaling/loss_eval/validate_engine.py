"""loss_eval の量子化順伝播 (Rust) を、対局用やねうら王の静的評価値と局面ごとに突き合わせる。

.psv から乱数で N 局面を選び、cshogi で SFEN にしてエンジンに `position sfen ...` → `e`
(USI 拡張コマンド: `Eval::evaluate()` の値を `eval = <v>` で表示) を送る。
エンジンの値は `clamp(raw / FV_SCALE, ±VALUE_MAX_EVAL)` (C++ の整数除算 = 0 方向への切り捨て)。
FV_SCALE=1 で生出力そのもの、FV_SCALE=16 (対局の既定値) で除算の扱いも確認する。

使い方:
    data/matchenv/bin/python experiments/009-data-scaling/loss_eval/validate_engine.py \\
        --net data/bulletou/checkpoints/009-p10-smoke/0001/nn.bin --psv <records.psv> --n 200
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cshogi
import numpy as np

HERE = Path(__file__).resolve().parent
TOOL = HERE / "target" / "release" / "loss_eval"
ENGINE = Path.home() / "engines" / "009-finny-avx2" / "YaneuraOu-by-gcc"
VALUE_MAX_EVAL = 32000 - 246 - 1  # VALUE_SUPERIOR (types.h, MAX_PLY = 246)


def engine_evals(engine: Path, evaldir: Path, sfens: list[str], fv_scale: int) -> list[int]:
    cmds = [
        "setoption name Threads value 1",
        f"setoption name EvalDir value {evaldir}",
        f"setoption name FV_SCALE value {fv_scale}",
        "isready",
    ]
    for s in sfens:
        cmds += [f"position sfen {s}", "e"]
    cmds.append("quit")
    out = subprocess.run(
        [str(engine)], input="\n".join(cmds) + "\n", capture_output=True, text=True, check=True
    ).stdout
    vals = [int(ln.split("=")[1]) for ln in out.splitlines() if ln.startswith("eval = ")]
    if len(vals) != len(sfens):
        raise SystemExit(
            f"engine returned {len(vals)} evals for {len(sfens)} positions:\n{out[-2000:]}"
        )
    return vals


def trunc_div(a: np.ndarray, b: int) -> np.ndarray:
    return (np.sign(a) * (np.abs(a) // b)).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--net", type=Path, required=True, help="nn.bin (その親ディレクトリを EvalDir にする)"
    )
    ap.add_argument("--psv", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--engine", type=Path, default=ENGINE)
    args = ap.parse_args()
    if args.net.name != "nn.bin":
        raise SystemExit("--net はファイル名 nn.bin (やねうら王は EvalDir/nn.bin を読む)")

    recs = np.fromfile(args.psv, dtype=np.uint8).reshape(-1, 40)
    idx = np.sort(
        np.random.default_rng(args.seed).choice(
            len(recs), size=min(args.n, len(recs)), replace=False
        )
    )
    sel = recs[idx]
    with tempfile.TemporaryDirectory() as td:
        sub = Path(td) / "sel.psv"
        sel.tofile(sub)
        subprocess.run(
            [
                str(TOOL),
                "eval",
                "--net",
                str(args.net),
                "--psv",
                str(sub),
                "--out",
                str(Path(td) / "o"),
                "--threads",
                "1",
            ],
            check=True,
            capture_output=True,
        )
        raw = np.load(Path(td) / "o.q.npy").astype(np.int64)

    board = cshogi.Board()
    sfens = []
    for r in sel:
        board.set_psfen(r[:32].copy())
        sfens.append(board.sfen())

    ok = True
    for fv in (1, 16):
        eng = np.array(engine_evals(args.engine, args.net.resolve().parent, sfens, fv), dtype=np.int64)
        exp = np.clip(trunc_div(raw, fv), -VALUE_MAX_EVAL, VALUE_MAX_EVAL)
        bad = np.flatnonzero(eng != exp)
        clipped = int((np.abs(trunc_div(raw, fv)) > VALUE_MAX_EVAL).sum())
        print(
            f"FV_SCALE={fv:2d}: {len(sfens)} positions, mismatches={bad.size}, "
            f"clipped={clipped}, |raw| max={int(np.abs(raw).max())}, eval range=[{eng.min()}, {eng.max()}]"
        )
        for b in bad[:5]:
            print(f"  mismatch: engine={eng[b]} tool={exp[b]} raw={raw[b]} sfen={sfens[b]}")
        ok &= bad.size == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
