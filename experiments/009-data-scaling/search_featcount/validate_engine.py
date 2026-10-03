"""計測版やねうら王 (FEATCOUNT) の HalfKA2 index を cshogi での再計算と突き合わせる。

互角局面集の局面から乱数で 0〜120 手進めた局面 (成駒・持ち駒・両手番を含む) を N 個作り、
エンジンの検証用コマンド `featsfen <sfen>` が返す両視点の index と、
feature_count/validate.py の halfka2_features() (HalfKA2 の定義式) の結果を比較する。
さらに featdump したヒストグラムが、Python 側で数えたヒストグラムと完全一致することを確認する。

使い方:
    data/matchenv/bin/python experiments/009-data-scaling/search_featcount/validate_engine.py \
        --engine ~/engines/009-featcount/YaneuraOu-by-gcc --n 2000
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import cshogi
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "feature_count"))
from validate import halfka2_features  # noqa: E402

DIM = 131949
REPO = HERE.parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(Path.home() / "engines/009-featcount/YaneuraOu-by-gcc"))
    ap.add_argument("--evaldir", default=str(Path.home() / "suisho11"))
    ap.add_argument("--book", default=str(REPO / "data/books/start_sfens_ply24.txt"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    book = [ln.strip().removeprefix("sfen ").strip() for ln in open(args.book) if ln.strip()]
    sfens: list[str] = []
    while len(sfens) < args.n:
        b = cshogi.Board(rng.choice(book))
        for _ in range(rng.randrange(0, 121)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if b.is_game_over():
            continue
        sfens.append(b.sfen())

    tmp = Path(tempfile.mkdtemp())
    npy = tmp / "hist.npy"
    cmds = [f"setoption name EvalDir value {args.evaldir}", "isready", "featreset"]
    cmds += [f"featsfen {s}" for s in sfens]
    cmds += [f"featdump {npy}", "quit"]
    out = subprocess.run(
        [args.engine], input="\n".join(cmds) + "\n", capture_output=True, text=True, check=True
    ).stdout
    lines = [ln for ln in out.splitlines() if ln.startswith("info string featsfen ")]
    assert len(lines) == len(sfens), (len(lines), len(sfens))

    hist = np.zeros(DIM, dtype=np.uint64)
    bad = 0
    for sfen, ln in zip(sfens, lines):
        body = ln.removeprefix("info string featsfen ")
        blk, wht = body.split(" white")
        got = {0: sorted(map(int, blk.split()[1:])), 1: sorted(map(int, wht.split()))}
        b = cshogi.Board(sfen)
        for persp in (0, 1):
            exp = sorted(halfka2_features(b, persp))
            np.add.at(hist, np.array(exp), 1)
            if got[persp] != exp or len(exp) != 40:
                bad += 1
                if bad <= 5:
                    print(f"MISMATCH persp={persp} {sfen}")
                    print("  engine-only", sorted(set(got[persp]) - set(exp)))
                    print("  python-only", sorted(set(exp) - set(got[persp])))
    dumped = np.load(npy)
    same = dumped.dtype == np.uint64 and dumped.shape == (DIM,) and np.array_equal(dumped, hist)
    stat = [ln for ln in out.splitlines() if "featdump path" in ln]
    print(stat[-1] if stat else "no featdump line")
    print(
        f"positions={len(sfens)} perspective mismatches={bad} "
        f"dump==python-hist: {same} dump sum={int(dumped.sum())} (expected {80 * len(sfens)})"
    )
    sys.exit(0 if bad == 0 and same else 1)


if __name__ == "__main__":
    main()
