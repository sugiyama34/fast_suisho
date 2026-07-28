"""フェーズ3 自己対局ハーネス: 学習ネット vs Suisho 11 (docs/PLAN.md の校正済み条件)。

- 同一エンジンバイナリで EvalDir だけ変えた 2 プロセスを起動
- 1 ペア = 同一開始局面で先後入替の 2 局。ペアごとに pentanomial SPRT を更新し、
  受理境界に達したら打ち切り (上限 --max-pairs)
- PLAN の校正条件を固定: `go movetime 240`・Threads 16・USI_Hash 1024・
  USI_Ponder false・MultiPV 1・NetworkDelay/NetworkDelay2 0・定跡なし・直列 1 局ずつ
- 終局判定: resign / 入玉宣言 (bestmove win) / cshogi の千日手判定 (連続王手含む) /
  --max-plies 到達で引き分け
- 出力: games/<name>/games.jsonl (1 局 1 行) + summary.json (SPRT 状態)。
  git 管理外 (experiments/**/games/ は gitignore 済み)

使い方:
    uv run python experiments/005-bulletou-sojo/match_runner.py \
        --candidate-evaldir data/bulletou/eval/main --name main-vs-suisho11
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cshogi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from match_sprt import PentanomialSprt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class UsiEngine:
    """最小限の USI エンジンドライバ (直列対局用・同期読み)。"""

    def __init__(
        self,
        binary: str,
        evaldir: str,
        threads: int,
        hash_mb: int,
        max_moves: int,
        stderr_path: Path | None = None,
    ):
        self.spawn_args = (binary, evaldir, threads, hash_mb, max_moves, stderr_path)
        stderr_fh = stderr_path.open("a") if stderr_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_fh,
            text=True,
            bufsize=1,
            cwd=str(Path(binary).parent),
        )
        self._send("usi")
        self._wait("usiok", timeout=30)
        for name, value in [
            ("EvalDir", evaldir),
            ("Threads", threads),
            ("USI_Hash", hash_mb),
            ("USI_Ponder", "false"),
            ("MultiPV", 1),
            ("NetworkDelay", 0),
            ("NetworkDelay2", 0),
            ("BookFile", "no_book"),
            ("MaxMovesToDraw", max_moves),
        ]:
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait("readyok", timeout=120)  # nn.bin 135MB のロード待ち

    def _send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait(self, token: str, timeout: float) -> None:
        deadline = time.time() + timeout
        assert self.proc.stdout is not None
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.strip() == token:
                return
        raise TimeoutError(f"engine did not answer {token!r} in {timeout}s")

    def newgame(self) -> None:
        self._send("usinewgame")
        self._send("isready")
        self._wait("readyok", timeout=30)

    def bestmove(self, position_cmd: str, movetime_ms: int) -> tuple[str, int | None]:
        """(bestmove, nodes)。nodes は最後の info 行から取る (無ければ None)。"""
        self._send(position_cmd)
        self._send(f"go movetime {movetime_ms}")
        nodes = None
        deadline = time.time() + movetime_ms / 1000.0 + 30.0
        assert self.proc.stdout is not None
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died during search")
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "info" and "nodes" in parts:
                try:
                    nodes = int(parts[parts.index("nodes") + 1])
                except (ValueError, IndexError):
                    pass
            if parts[0] == "bestmove":
                return parts[1], nodes
        raise TimeoutError("no bestmove")

    def quit(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def play_game(
    cand: UsiEngine,
    base: UsiEngine,
    start_sfen: str,
    cand_is_black: bool,
    movetime_ms: int,
    max_plies: int,
) -> tuple[float, str, int]:
    """1 局対局し (候補側得点 1/0.5/0, 終局理由, 手数) を返す。

    start_sfen は 24 手目 (後手番) の局面なので、初手を指すのは後手側。
    """
    for e in (cand, base):
        e.newgame()
    board = cshogi.Board(sfen=start_sfen)
    moves: list[str] = []
    while True:
        if board.is_game_over():  # 詰み (合法手なし): 手番側の負け
            stm_is_cand = (board.turn == cshogi.BLACK) == cand_is_black
            return (0.0 if stm_is_cand else 1.0), "mate", len(moves)
        draw = board.is_draw()
        if draw == cshogi.REPETITION_DRAW:
            return 0.5, "sennichite", len(moves)
        if draw in (cshogi.REPETITION_WIN, cshogi.REPETITION_SUPERIOR):
            # 手番側の勝ち扱い (連続王手の千日手等は cshogi の判定に従う)
            stm_is_cand = (board.turn == cshogi.BLACK) == cand_is_black
            return (1.0 if stm_is_cand else 0.0), "repetition_win", len(moves)
        if draw in (cshogi.REPETITION_LOSE, cshogi.REPETITION_INFERIOR):
            stm_is_cand = (board.turn == cshogi.BLACK) == cand_is_black
            return (0.0 if stm_is_cand else 1.0), "repetition_lose", len(moves)
        if len(moves) >= max_plies:
            return 0.5, "max_plies", len(moves)

        stm_is_cand = (board.turn == cshogi.BLACK) == cand_is_black
        engine = cand if stm_is_cand else base
        pos = f"position sfen {start_sfen}"
        if moves:
            pos += " moves " + " ".join(moves)
        mv, _ = engine.bestmove(pos, movetime_ms)
        if mv == "resign":
            return (0.0 if stm_is_cand else 1.0), "resign", len(moves)
        if mv == "win":  # 入玉宣言
            return (1.0 if stm_is_cand else 0.0), "declare_win", len(moves)
        move_obj = board.move_from_usi(mv)
        if move_obj not in board.legal_moves:
            # 不正着手は即負け (ログで発覚できるよう理由を分ける)
            return (0.0 if stm_is_cand else 1.0), f"illegal_move:{mv}", len(moves)
        board.push(move_obj)
        moves.append(mv)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--candidate-evaldir", required=True, help="学習ネット nn.bin のあるディレクトリ"
    )
    ap.add_argument("--baseline-evaldir", default="/home/sugiyama/suisho11")
    ap.add_argument("--engine", default="/home/sugiyama/YaneuraOu/source/YaneuraOu-by-gcc")
    ap.add_argument("--book", default=str(HERE / "start_sfens_500.txt"))
    ap.add_argument("--name", required=True, help="出力ディレクトリ名 (games/<name>/)")
    ap.add_argument("--movetime", type=int, default=240)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--hash-mb", type=int, default=1024)
    ap.add_argument("--max-pairs", type=int, default=500)
    ap.add_argument("--max-plies", type=int, default=320)
    ap.add_argument("--elo0", type=float, default=-30.0)
    ap.add_argument("--elo1", type=float, default=0.0)
    ap.add_argument("--start-pair", type=int, default=0, help="再開用: このペア番号から始める")
    args = ap.parse_args()

    out_dir = HERE / "games" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    games_path = out_dir / "games.jsonl"
    summary_path = out_dir / "summary.json"

    with open(args.book) as fh:
        sfens = [line.strip().removeprefix("sfen ").strip() for line in fh if line.strip()]
    sfens = sfens[: args.max_pairs]

    sprt = PentanomialSprt(elo0=args.elo0, elo1=args.elo1)
    # 再開: 既存 games.jsonl から完結ペア (2 局揃い) だけ再構築し、ファイルも圧縮し直す
    done_recs: dict[int, list[dict]] = {}
    if games_path.exists():
        with games_path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                done_recs.setdefault(rec["pair"], []).append(rec)
        done_recs = {p: recs[:2] for p, recs in done_recs.items() if len(recs) >= 2}
        with games_path.open("w") as fh:
            for p in sorted(done_recs):
                for rec in done_recs[p]:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for recs in done_recs.values():
            sprt.add_pair(sum(r["cand_score"] for r in recs))

    def spawn() -> tuple[UsiEngine, UsiEngine]:
        return (
            UsiEngine(
                args.engine,
                args.candidate_evaldir,
                args.threads,
                args.hash_mb,
                args.max_plies,
                stderr_path=out_dir / "engine-cand.stderr.log",
            ),
            UsiEngine(
                args.engine,
                args.baseline_evaldir,
                args.threads,
                args.hash_mb,
                args.max_plies,
                stderr_path=out_dir / "engine-base.stderr.log",
            ),
        )

    cand, base = spawn()
    games_fh = games_path.open("a")
    try:
        for i, sfen in enumerate(sfens):
            if i < args.start_pair or i in done_recs:
                continue
            recs: list[dict] = []
            for attempt in (1, 2):
                try:
                    recs = []
                    for cand_is_black in (True, False):
                        score, reason, plies = play_game(
                            cand, base, sfen, cand_is_black, args.movetime, args.max_plies
                        )
                        recs.append(
                            {
                                "pair": i,
                                "cand_is_black": cand_is_black,
                                "cand_score": score,
                                "reason": reason,
                                "plies": plies,
                                "sfen": sfen,
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            }
                        )
                    break
                except (RuntimeError, TimeoutError) as err:
                    print(
                        f"pair {i + 1} attempt {attempt} failed: {err}; respawning engines",
                        flush=True,
                    )
                    for e in (cand, base):
                        e.quit()
                    time.sleep(3)
                    cand, base = spawn()
                    recs = []
            if len(recs) != 2:
                print(f"pair {i + 1}: skipped after repeated engine failure", flush=True)
                continue
            pair_score = sum(r["cand_score"] for r in recs)
            for rec in recs:
                games_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            games_fh.flush()
            sprt.add_pair(pair_score)
            summary = sprt.summary()
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
            print(
                f"pair {i + 1}: score={pair_score} | n={summary['pairs']} "
                f"elo={summary['elo']} ci={summary['elo_ci95']} llr={summary['llr']} "
                f"{summary['sprt']}",
                flush=True,
            )
            if summary["sprt"] != "continue":
                print(f"SPRT decision: {summary['sprt']}", flush=True)
                break
    finally:
        cand.quit()
        base.quit()
        games_fh.close()
    print(json.dumps(sprt.summary(), ensure_ascii=False))


if __name__ == "__main__":
    main()
