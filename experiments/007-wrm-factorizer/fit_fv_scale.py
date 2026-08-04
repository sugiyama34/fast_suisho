"""per-net FV_SCALE 推定: 共通局面での評価値分布を Suisho 11 に線形フィットする。

方法 (たややん氏の提案 2026-08-04 に基づく):
- floodgate.hcpe から等間隔サンプルした局面を、各評価関数で `go depth 1` 評価
  (全 net 共通で FV_SCALE=16, Threads=1, 定跡なし)
- Suisho 11 の cp が [30, 1500] の局面に絞り、slope = median(cp_net / cp_suisho)
  (原点通過 Theil-Sen)。net の適正 FV_SCALE = slope × 40
  (アンカー: 水匠評価関数の適正値 40 = たややん氏の推定)

使い方 (matchenv の python で):
    data/matchenv/bin/python experiments/007-wrm-factorizer/fit_fv_scale.py --n 10000
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import cshogi
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ENGINE = "/home/sugiyama/YaneuraOu/source/YaneuraOu-by-gcc"
HCPE = REPO / "data" / "teacher" / "floodgate" / "floodgate.hcpe"

NETS = {
    "suisho11": "/home/sugiyama/suisho11",
    "007-wrm-ep20": str(REPO / "data/bulletou/checkpoints/007-wrm/0020"),
    "007-ctrl-ep20": str(REPO / "data/bulletou/checkpoints/007-ctrl/0020"),
    "006-yane-ep16": str(REPO / "data/bulletou/checkpoints/006-yane20/0016"),
    "007-wrmshot-ep20": str(REPO / "data/bulletou/checkpoints/007-wrmshot/0020"),
    "007-lr1shot-ep20": str(REPO / "data/bulletou/checkpoints/007-lr1shot/0020"),
}
ANCHOR_NET = "suisho11"
ANCHOR_FV = 40  # たややん氏の推定 (水匠評価関数の適正値)


def sample_sfens(n: int) -> list[str]:
    dt = np.dtype(cshogi.HuffmanCodedPosAndEval)
    data = np.fromfile(HCPE, dtype=dt)
    stride = max(1, len(data) // n)
    board = cshogi.Board()
    sfens = []
    for rec in data[::stride][:n]:
        board.set_hcp(rec["hcp"])
        sfens.append(board.sfen())
    return sfens


def eval_positions(evaldir: str, sfens: list[str]) -> list[int | None]:
    proc = subprocess.Popen(
        [ENGINE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        cwd=str(Path(ENGINE).parent),
    )

    def send(s: str) -> None:
        proc.stdin.write(s + "\n")
        proc.stdin.flush()

    def wait_for(tok: str, timeout: float = 120.0) -> list[str]:
        lines = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            lines.append(line)
            if line.startswith(tok):
                return lines
        raise TimeoutError(tok)

    send("usi")
    wait_for("usiok")
    for k, v in [
        ("EvalDir", evaldir),
        ("Threads", 1),
        ("USI_Hash", 128),
        ("BookFile", "no_book"),
        ("NetworkDelay", 0),
        ("NetworkDelay2", 0),
        ("FV_SCALE", 16),  # 全 net 共通の固定除数 (生スケール比較のため)
    ]:
        send(f"setoption name {k} value {v}")
    send("isready")
    wait_for("readyok")

    out: list[int | None] = []
    for sfen in sfens:
        send(f"position sfen {sfen}")
        send("go depth 1")
        cp = None
        for line in wait_for("bestmove", timeout=30):
            parts = line.split()
            if parts[:1] == ["info"] and "score" in parts:
                i = parts.index("score")
                if parts[i + 1] == "cp":
                    cp = int(parts[i + 2])
                else:  # mate 等は除外
                    cp = None
        out.append(cp)
    send("quit")
    proc.wait(timeout=10)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--band", type=float, nargs=2, default=(30, 1500))
    args = ap.parse_args()

    sfens = sample_sfens(args.n)
    print(f"positions: {len(sfens)} (from {HCPE.name})")

    evals: dict[str, list[int | None]] = {}
    for name, evaldir in NETS.items():
        t0 = time.time()
        evals[name] = eval_positions(evaldir, sfens)
        n_ok = sum(1 for v in evals[name] if v is not None)
        print(f"{name}: {n_ok}/{len(sfens)} evals in {time.time() - t0:.0f}s", flush=True)

    ref = evals[ANCHOR_NET]
    lo, hi = args.band
    result = {}
    for name in NETS:
        if name == ANCHOR_NET:
            result[name] = {"slope": 1.0, "fv_scale": ANCHOR_FV, "n_used": None}
            continue
        ratios = [
            v / r
            for v, r in zip(evals[name], ref)
            if v is not None and r is not None and lo <= abs(r) <= hi and abs(v) <= 30000
        ]
        slope = statistics.median(ratios)
        result[name] = {
            "slope": round(slope, 4),
            "fv_scale": round(slope * ANCHOR_FV, 1),
            "n_used": len(ratios),
        }
        print(
            f"{name}: slope={slope:.4f} (n={len(ratios)}) -> 適正 FV_SCALE ≈ {slope * ANCHOR_FV:.1f}"
        )

    out = HERE / "fv_scale_fit.json"
    out.write_text(
        json.dumps(
            {
                "anchor": {ANCHOR_NET: ANCHOR_FV},
                "band_cp": list(args.band),
                "n_positions": len(sfens),
                "fit": result,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
