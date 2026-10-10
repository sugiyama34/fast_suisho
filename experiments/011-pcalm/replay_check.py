"""experiment-011: kajiki のエンジンが experiment-009 (suzuki) と同じ指し手を返すかの確認 (hypothesis.md §4.5)。

experiment-009 の `final-full-e16@48-vs-s11@32-300k-2000p` の棋譜から先頭 N ペア (pair 0..N−1, 先後 2 局ずつ) を、
同じエンジン・同じネット・同じ設定 (300k ノード, Threads 1, Hash 256 MB, FV_SCALE 48 / 32) で指し直し、
全局の指し手列と結果が一致するかを数える。match_nodes.py と同じ部品 (NodesEngine, play_game) を使う。

使い方 (kajiki, スタックは 8 MB):
    ulimit -s 8192; data/matchenv/bin/python experiments/011-pcalm/replay_check.py --pairs 50 --workers 4
出力: <out>/replay.jsonl (1 局 1 行) と <out>/summary.json
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "005-bulletou-sojo"))
sys.path.insert(0, str(REPO / "experiments" / "009-data-scaling"))
from match_nodes import NodesEngine  # noqa: E402
from match_runner import play_game  # noqa: E402

GAME = "final-full-e16@48-vs-s11@32-300k-2000p"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--games", type=Path, default=REPO / "experiments" / "009-data-scaling" / "games" / GAME
    )
    ap.add_argument(
        "--engine", default=str(Path.home() / "engines" / "009-finny-avx2" / "YaneuraOu-by-gcc")
    )
    ap.add_argument("--candidate-evaldir", default="/mnt/D/sugiyama/checkpoints/009-full/0016")
    ap.add_argument("--baseline-evaldir", default=str(Path.home() / "suisho11"))
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("/mnt/D/sugiyama/011/replay"))
    args = ap.parse_args()

    summ = json.loads((args.games / "summary.json").read_text())
    ref = {}
    for line in (args.games / "games.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["pair"] < args.pairs:
            ref[(r["pair"], r["cand_is_black"])] = r
    assert len(ref) == 2 * args.pairs, f"{len(ref)} reference games"
    args.out.mkdir(parents=True, exist_ok=True)
    nodes, max_plies = summ["nodes"], 320
    todo: queue.Queue[int] = queue.Queue()
    for p in range(args.pairs):
        todo.put(p)
    lock = threading.Lock()
    out_fh = (args.out / "replay.jsonl").open("w")
    rows: list[dict] = []

    def worker(w: int) -> None:
        mk = lambda d, fv, tag: NodesEngine(  # noqa: E731
            args.engine, d, 1, summ["hash_mb"], max_plies,
            stderr_path=args.out / f"engine-{tag}-w{w}.stderr.log", fv_scale=fv,
        )  # fmt: skip
        cand = mk(args.candidate_evaldir, summ["candidate_fv_scale"], "cand")
        base = mk(args.baseline_evaldir, summ["baseline_fv_scale"], "base")
        try:
            while True:
                try:
                    p = todo.get_nowait()
                except queue.Empty:
                    return
                for black in (True, False):
                    r0 = ref[(p, black)]
                    moves: list[str] = []
                    score, reason, plies = play_game(
                        cand, base, r0["sfen"], black, nodes, max_plies, moves_out=moves
                    )
                    row = {
                        "pair": p,
                        "cand_is_black": black,
                        "same_moves": " ".join(moves) == r0["moves"],
                        "same_score": score == r0["cand_score"],
                        "plies": plies,
                        "ref_plies": r0["plies"],
                        "first_diff_ply": next(
                            (
                                i
                                for i, (a, b) in enumerate(zip(moves, r0["moves"].split()))
                                if a != b
                            ),
                            None,
                        ),
                    }
                    with lock:
                        rows.append(row)
                        out_fh.write(json.dumps(row) + "\n")
                        out_fh.flush()
                        print(len(rows), row, flush=True)
        finally:
            cand.quit()
            base.quit()

    ts = [threading.Thread(target=worker, args=(w,)) for w in range(args.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    res = {
        "reference": str(args.games),
        "engine": args.engine,
        "games": len(rows),
        "same_moves": sum(r["same_moves"] for r in rows),
        "same_score": sum(r["same_score"] for r in rows),
        "pairs": args.pairs,
    }
    (args.out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
