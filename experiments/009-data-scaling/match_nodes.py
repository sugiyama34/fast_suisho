"""experiment-009 の対局ハーネス: 固定ノード数 × 並列対局 (学習ネット vs 基準ネット)。

experiment-005 の match_runner.py (movetime・16 スレッド・直列) からの変更点:
- 探索制限を `go nodes N` (固定ノード数) にする。全ネットは同一アーキテクチャで
  NPS が等しいため movetime である必要がなく、CPU 競合 (学習の loader と同時実行) で
  結果が変わらない (2026-09-28 ユーザー決定)
- 各エンジン Threads=1 で探索が決定的になる。代わりに W 組のエンジンペアを並列に走らせる
- 開始局面は互角局面集 (ply 24, 30,053 局面) から等間隔ストライドで --pairs 個。
  全 arm で同じ局面集合を使う (arm 間比較の分散を減らす)

対局ルール・終局判定・pentanomial 統計は experiment-005 の実装をそのまま再利用する
(`play_game`, `UsiEngine`, `PentanomialSprt`)。

出力: <games-dir>/<name>/games.jsonl (1 局 1 行) + summary.json。再実行で完結ペアを再利用する。

使い方:
    nice -n 10 data/matchenv/bin/python experiments/009-data-scaling/match_nodes.py \
        --candidate-evaldir data/bulletou/checkpoints/009-full/0020 --name full-e20 \
        --nodes 1000000 --pairs 500 --workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "005-bulletou-sojo"))

from match_runner import UsiEngine, play_game  # noqa: E402
from match_sprt import PentanomialSprt  # noqa: E402


class NodesEngine(UsiEngine):
    """bestmove の制限を `go nodes N` に差し替えた UsiEngine。

    play_game は第 2 引数を movetime_ms として渡してくるが、ここではノード数として解釈する。
    """

    def bestmove(self, position_cmd: str, nodes: int) -> tuple[str, int | None]:
        self._send(position_cmd)
        self._send(f"go nodes {nodes}")
        got = None
        deadline = time.time() + 300.0  # 1 スレッド・高負荷時でも十分な上限
        while time.time() < deadline:
            parts = self._readline(deadline).split()
            if not parts:
                continue
            if parts[0] == "info" and "nodes" in parts:
                try:
                    got = int(parts[parts.index("nodes") + 1])
                except (ValueError, IndexError):
                    pass
            if parts[0] == "bestmove":
                return parts[1], got
        raise TimeoutError("no bestmove")


def load_positions(book: Path, n_pairs: int) -> list[str]:
    with book.open() as fh:
        sfens = [ln.strip().removeprefix("sfen ").strip() for ln in fh if ln.strip()]
    stride = len(sfens) // n_pairs
    return [sfens[i * stride] for i in range(n_pairs)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-evaldir", required=True)
    ap.add_argument("--baseline-evaldir", default=str(Path.home() / "suisho11"))
    ap.add_argument(
        "--engine", default=str(Path.home() / "YaneuraOu" / "source" / "YaneuraOu-by-gcc")
    )
    ap.add_argument("--book", default=str(REPO / "data" / "books" / "start_sfens_ply24.txt"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--games-dir", default=str(HERE / "games"))
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--pairs", type=int, default=500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--hash-mb", type=int, default=256)
    ap.add_argument("--max-plies", type=int, default=320)
    args = ap.parse_args()

    out_dir = Path(args.games_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    games_path = out_dir / "games.jsonl"
    summary_path = out_dir / "summary.json"
    positions = load_positions(Path(args.book), args.pairs)

    # 再開: 完結ペア (2 局) のみ採用
    done: dict[int, list[dict]] = {}
    if games_path.exists():
        for line in games_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.setdefault(rec["pair"], []).append(rec)
        done = {p: r[:2] for p, r in done.items() if len(r) >= 2}
        # 局面はストライド (= --pairs 依存) で選ぶので、--pairs を変えて再開すると
        # 同じ pair 番号が別局面を指す。混ざらないよう停止する
        bad = [p for p, r in done.items() if p >= len(positions) or r[0]["sfen"] != positions[p]]
        if bad:
            raise SystemExit(f"{games_path}: pairs {bad[:5]} do not match --pairs {args.pairs}")
        tmp = games_path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as fh:
            for p in sorted(done):
                for rec in done[p]:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, games_path)

    sprt = PentanomialSprt()
    for recs in done.values():
        sprt.add_pair(sum(r["cand_score"] for r in recs))

    todo: queue.Queue[int] = queue.Queue()
    for i in range(len(positions)):
        if i not in done:
            todo.put(i)
    lock = threading.Lock()
    games_fh = games_path.open("a")
    failed: list[int] = []
    meta = {
        "candidate_evaldir": str(Path(args.candidate_evaldir).resolve()),
        "baseline_evaldir": args.baseline_evaldir,
        "engine": args.engine,
        "nodes": args.nodes,
        "threads_per_engine": 1,
        "hash_mb": args.hash_mb,
        "pairs_target": args.pairs,
        "book": args.book,
    }

    def spawn(w: int) -> tuple[NodesEngine, NodesEngine]:
        mk = lambda evaldir, tag: NodesEngine(  # noqa: E731
            args.engine,
            evaldir,
            1,
            args.hash_mb,
            args.max_plies,
            stderr_path=out_dir / f"engine-{tag}-w{w}.stderr.log",
        )
        return mk(args.candidate_evaldir, "cand"), mk(args.baseline_evaldir, "base")

    def worker(w: int) -> None:
        cand, base = spawn(w)
        try:
            while True:
                try:
                    i = todo.get_nowait()
                except queue.Empty:
                    return
                recs: list[dict] = []
                for attempt in (1, 2, 3):
                    try:
                        recs = []
                        for cand_is_black in (True, False):
                            moves: list[str] = []
                            t0 = time.time()
                            score, reason, plies = play_game(
                                cand,
                                base,
                                positions[i],
                                cand_is_black,
                                args.nodes,
                                args.max_plies,
                                moves_out=moves,
                            )
                            recs.append(
                                {
                                    "pair": i,
                                    "cand_is_black": cand_is_black,
                                    "cand_score": score,
                                    "reason": reason,
                                    "plies": plies,
                                    "sfen": positions[i],
                                    "moves": " ".join(moves),
                                    "sec": round(time.time() - t0, 2),
                                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                }
                            )
                        break
                    except (RuntimeError, TimeoutError) as err:
                        print(f"[w{w}] pair {i} attempt {attempt}: {err}; respawn", flush=True)
                        for e in (cand, base):
                            e.quit()
                        time.sleep(2)
                        cand, base = spawn(w)
                        recs = []
                with lock:
                    if len(recs) != 2:
                        failed.append(i)
                        continue
                    for rec in recs:
                        games_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    games_fh.flush()
                    sprt.add_pair(sum(r["cand_score"] for r in recs))
                    s = {**sprt.summary(), "failed_pairs": sorted(failed), **meta}
                    summary_path.write_text(json.dumps(s, ensure_ascii=False, indent=1))
                    if sprt.n_pairs % 20 == 0:
                        print(f"n={s['pairs']} elo={s['elo']} ci={s['elo_ci95']}", flush=True)
        finally:
            cand.quit()
            base.quit()

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    games_fh.close()
    s = {**sprt.summary(), "failed_pairs": sorted(failed), **meta}
    summary_path.write_text(json.dumps(s, ensure_ascii=False, indent=1))
    print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
