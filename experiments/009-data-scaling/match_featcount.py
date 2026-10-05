"""experiment-009: 水匠 11 同士の固定ノード対局で、探索中の NNUE 評価ごとの HalfKA2 特徴量の発火回数を集計する。

計測版やねうら王 (`~/engines/009-featcount/`, `-DFEATCOUNT` ビルド) を使う。エンジンは
ネットワークを実際に計算した評価 (探索中の evaluate() のうち computed_score キャッシュに
当たらなかったもの) ごとに、両視点の active index (40 + 40) を数える。USI 拡張:
`featdump <path>` (u64[131949] の .npy), `featreset`, `featstat`。詳細は
search_featcount/README.md。

- 1 開始局面 1 局 (同一エンジン同士の固定ノード・Threads 1 は決定的で、先後を入れ替えても
  同じ棋譜になるため)。開始局面は match_nodes.py と同じ等間隔ストライド (--openings 個)
- 対局手順は match_nodes.py と同じ (エンジン 2 プロセス、毎局 isready)。
  したがって棋譜は match_nodes.py の cand_is_black=True の局と一致する
- 出力 (<out-dir>/<name>/, 既定 games/featcount/<name>/):
  - search.npy: 探索中の評価の発火回数 (両視点の合計, u64[131949])
  - played.npy: 棋譜上の局面 (開始局面 + 各手の後, 終局局面を含む) の発火回数 (cshogi で計算)
  - games.jsonl: 1 局 1 行 (棋譜, 評価回数, ノード数)
  - summary.json
- 再開: summary.json の games_done に含まれる局だけを完了として扱い、npy もその時点のものを使う

使い方:
    nice -n 10 data/matchenv/bin/python experiments/009-data-scaling/match_featcount.py \
        --name smoke-300k --openings 20 --nodes 300000 --workers 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import cshogi
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "005-bulletou-sojo"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "feature_count"))

from match_nodes import NodesEngine, load_positions  # noqa: E402
from match_runner import play_game  # noqa: E402
from validate import halfka2_features  # noqa: E402

DIM = 131949


class FeatEngine(NodesEngine):
    """NodesEngine + 探索ノード数の積算 + featdump/featreset。"""

    nodes_total = 0

    def bestmove(self, position_cmd: str, nodes: int) -> tuple[str, int | None]:
        mv, got = super().bestmove(position_cmd, nodes)
        self.nodes_total += got or 0
        return mv, got

    def take(self, path: Path) -> tuple[np.ndarray, int]:
        """発火回数を dump して読み込み、エンジン側を 0 に戻す。(hist, 評価回数) を返す。"""
        self._send(f"featdump {path}")
        deadline = time.time() + 60
        while True:
            parts = self._readline(deadline).split()
            if parts[:3] == ["info", "string", "featdump"]:
                break
        kv = dict(zip(parts[5::2], parts[6::2]))
        evals, total, bad = int(kv["evals"]), int(kv["sum"]), int(kv["bad"])
        hist = np.load(path)
        if bad or total != 80 * evals or int(hist.sum()) != total:
            raise RuntimeError(f"featdump inconsistent: {' '.join(parts)}")
        self._send("featreset")
        self._wait("featreset", timeout=60)
        return hist, evals

    def _wait(self, token: str, timeout: float) -> None:
        deadline = time.time() + timeout
        while token not in self._readline(deadline).split():
            pass


def played_hist(start_sfen: str, moves: list[str]) -> np.ndarray:
    """棋譜上の全局面 (開始局面 + 各手の後) の両視点の発火回数。"""
    h = np.zeros(DIM, dtype=np.uint64)
    b = cshogi.Board(start_sfen)
    idx: list[int] = []
    for i in range(len(moves) + 1):
        if i:
            b.push_usi(moves[i - 1])
        idx += halfka2_features(b, 0) + halfka2_features(b, 1)
    np.add.at(h, np.array(idx), 1)
    return h


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default=str(Path.home() / "engines/009-featcount/YaneuraOu-by-gcc"))
    ap.add_argument("--evaldir", default=str(Path.home() / "suisho11"))
    ap.add_argument("--book", default=str(REPO / "data/books/start_sfens_ply24.txt"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default=str(HERE / "games" / "featcount"))
    ap.add_argument("--nodes", type=int, default=300_000)
    ap.add_argument("--openings", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hash-mb", type=int, default=256)
    ap.add_argument("--max-plies", type=int, default=320)
    ap.add_argument("--save-every", type=int, default=20, help="この局数ごとに npy/summary を保存")
    args = ap.parse_args()

    out = Path(args.out_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    games_path, summary_path = out / "games.jsonl", out / "summary.json"
    search_path, played_path = out / "search.npy", out / "played.npy"
    positions = load_positions(Path(args.book), args.openings)

    meta = {
        "engine": args.engine,
        "engine_sha256": sha256(args.engine),
        "evaldir": args.evaldir,
        "nn_bin_sha256": sha256(str(Path(args.evaldir) / "nn.bin")),
        "nodes": args.nodes,
        "threads_per_engine": 1,
        "hash_mb": args.hash_mb,
        "max_plies": args.max_plies,
        "openings_target": args.openings,
        "book": args.book,
        "played_def": "開始局面 + 各手の後の局面 (終局局面を含む)",
        "search_def": "探索中 evaluate() がネットワークを実際に計算した回数 (両視点, 1 評価 80)",
    }

    # 再開
    search = np.zeros(DIM, dtype=np.uint64)
    played = np.zeros(DIM, dtype=np.uint64)
    recs: dict[int, dict] = {}
    if summary_path.exists() and search_path.exists():
        prev = json.loads(summary_path.read_text())
        for k in ("nodes", "openings_target", "engine_sha256", "nn_bin_sha256"):
            if prev[k] != meta[k]:
                raise SystemExit(f"resume mismatch on {k}: {prev[k]} != {meta[k]}")
        done_set = set(prev["games_done"])
        for line in games_path.read_text().splitlines():
            r = json.loads(line)
            if r["opening"] in done_set:
                recs[r["opening"]] = r
        assert set(recs) == done_set
        search, played = np.load(search_path), np.load(played_path)
    with games_path.open("w") as fh:
        for i in sorted(recs):
            fh.write(json.dumps(recs[i], ensure_ascii=False) + "\n")

    todo: queue.Queue[int] = queue.Queue()
    for i in range(len(positions)):
        if i not in recs:
            todo.put(i)
    lock = threading.Lock()
    games_fh = games_path.open("a")
    failed: list[int] = []
    t_start = time.time()

    def summary() -> dict:
        n = len(recs)
        ev = sum(r["evals"] for r in recs.values())
        nd = sum(r["nodes"] for r in recs.values())
        pl = sum(r["plies"] for r in recs.values())
        reasons: dict[str, int] = {}
        for r in recs.values():
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        return {
            **meta,
            "games": n,
            "evaluations": ev,
            "evaluations_per_game": ev / n if n else None,
            "search_nodes": nd,
            "evaluations_per_node": ev / nd if nd else None,
            "plies": pl,
            "played_positions": pl + n,
            "search_hist_sum": int(search.sum()),
            "played_hist_sum": int(played.sum()),
            "reasons": reasons,
            "failed_openings": sorted(failed),
            "games_done": sorted(recs),
            "elapsed_sec_this_run": round(time.time() - t_start, 1),
        }

    def checkpoint() -> None:
        games_fh.flush()
        save_npy(search_path, search)
        save_npy(played_path, played)
        tmp = summary_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary(), ensure_ascii=False, indent=1))
        os.replace(tmp, summary_path)

    tmpdir = Path(tempfile.mkdtemp(prefix="featcount-", dir=out))

    def spawn(w: int) -> tuple[FeatEngine, FeatEngine]:
        mk = lambda tag: FeatEngine(  # noqa: E731
            args.engine,
            args.evaldir,
            1,
            args.hash_mb,
            args.max_plies,
            stderr_path=out / f"engine-{tag}-w{w}.stderr.log",
        )
        a, b = mk("a"), mk("b")
        for e in (a, b):  # 起動時の評価 (isready 等) を捨てる
            e.take(tmpdir / f"w{w}.npy")
        return a, b

    def worker(w: int) -> None:
        a, b = spawn(w)
        try:
            while True:
                try:
                    i = todo.get_nowait()
                except queue.Empty:
                    return
                rec = None
                for attempt in (1, 2, 3):
                    try:
                        for e in (a, b):
                            e.nodes_total = 0
                        moves: list[str] = []
                        t0 = time.time()
                        score, reason, plies = play_game(
                            a, b, positions[i], True, args.nodes, args.max_plies, moves_out=moves
                        )
                        sec = time.time() - t0
                        ha, ea = a.take(tmpdir / f"w{w}.npy")
                        hb, eb = b.take(tmpdir / f"w{w}.npy")
                        rec = {
                            "opening": i,
                            "sfen": positions[i],
                            "moves": " ".join(moves),
                            "plies": plies,
                            "reason": reason,
                            "black_score": score,
                            "evals": ea + eb,
                            "nodes": a.nodes_total + b.nodes_total,
                            "sec": round(sec, 2),
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                        hg = ha + hb
                        break
                    except (RuntimeError, TimeoutError) as err:
                        print(f"[w{w}] opening {i} attempt {attempt}: {err}; respawn", flush=True)
                        for e in (a, b):
                            e.quit()
                        time.sleep(2)
                        a, b = spawn(w)
                        rec = None
                hp = played_hist(positions[i], rec["moves"].split()) if rec else None
                with lock:
                    if rec is None:
                        failed.append(i)
                        continue
                    search[:] += hg
                    played[:] += hp
                    recs[i] = rec
                    games_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if len(recs) % args.save_every == 0:
                        checkpoint()
                        print(
                            f"games={len(recs)} evals/game={summary()['evaluations_per_game']:.0f}",
                            flush=True,
                        )
        finally:
            a.quit()
            b.quit()

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with lock:
        checkpoint()
    games_fh.close()
    for f in tmpdir.iterdir():
        f.unlink()
    tmpdir.rmdir()
    s = summary()
    s.pop("games_done")
    print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
