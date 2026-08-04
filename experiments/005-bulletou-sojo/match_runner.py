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
import os
import queue
import subprocess
import sys
import threading
import time
from collections import Counter, deque
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
        fv_scale: int | None = None,
    ):
        self.spawn_args = (binary, evaldir, threads, hash_mb, max_moves, stderr_path, fv_scale)
        # silent crash 調査用 (repo issue 参照): どちら側が・どの exit code で・
        # 何を最後に出力して死んだかを記録する。挙動には影響しない
        self.label = "cand" if stderr_path and "cand" in stderr_path.name else "base"
        self.recent_stdout: deque[str] = deque(maxlen=8)
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
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._start_reader()
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
            # 手数上限引き分けはハーネス側 (len(moves) >= max_plies) でのみ判定する。
            # エンジン側 MaxMovesToDraw は gamePly 基準 (開始 sfen の手数 ~24 を含む)
            # でハーネスの手数と食い違うため無効化 (0)
            ("MaxMovesToDraw", 0),
        ]:
            self._send(f"setoption name {name} value {value}")
        # FV_SCALE: 評価関数ごとに適正値が異なる (issue #18 とは別件、たややん氏
        # 指摘 2026-08-04)。未指定ならエンジン default (16) のまま = 005-007 の従来条件
        if fv_scale is not None:
            self._send(f"setoption name FV_SCALE value {fv_scale}")
        self._send("isready")
        self._wait("readyok", timeout=120)  # nn.bin 135MB のロード待ち

    def _send(self, line: str) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except OSError as err:
            # エンジンがアイドル中に死んでいた場合、書き込み側で BrokenPipeError が
            # 出る。呼び出し元の respawn リトライに乗せるため RuntimeError に揃える
            raise RuntimeError(f"engine died ({self._death_info()}; write failed: {err})") from err

    def _death_info(self) -> str:
        """死亡時の診断情報 (silent crash 調査用)。exit code が負なら signal 死、
        0 なら自発 exit (プロトコル系) の可能性が高い。"""
        tail = " | ".join(s.strip() for s in list(self.recent_stdout)[-3:])
        return f"side={self.label} exit_code={self.proc.poll()} last_stdout=[{tail}]"

    def _start_reader(self) -> None:
        """stdout をスレッドで読み queue に流す。blocking readline だと
        エンジンが無出力でハングした際にタイムアウトを検出できないため。"""
        assert self.proc.stdout is not None

        def pump(stdout=self.proc.stdout, q=self._lines, recent=self.recent_stdout) -> None:
            for line in stdout:
                recent.append(line)
                q.put(line)
            q.put(None)  # EOF

        threading.Thread(target=pump, daemon=True).start()

    def _readline(self, deadline: float) -> str:
        try:
            line = self._lines.get(timeout=max(0.0, deadline - time.time()))
        except queue.Empty:
            raise TimeoutError("engine produced no output before deadline") from None
        if line is None:
            # exit code 確定を少し待つ (EOF 直後は reap 前で None になり得る)
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            raise RuntimeError(f"engine died ({self._death_info()})")
        return line

    def _wait(self, token: str, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._readline(deadline).strip() == token:
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
        while time.time() < deadline:
            parts = self._readline(deadline).split()
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
    nodes_out: dict[str, list[int]] | None = None,
) -> tuple[float, str, int]:
    """1 局対局し (候補側得点 1/0.5/0, 終局理由, 手数) を返す。

    start_sfen は 24 手目 (後手番) の局面なので、初手を指すのは後手側。
    """
    for e in (cand, base):
        e.newgame()
    board = cshogi.Board(sfen=start_sfen)
    moves: list[str] = []
    # 千日手判定: 公式ルール通り「同一局面 4 回目」で成立とする。cshogi の
    # is_draw() は初回の再訪で REPETITION_* を返すため、それをそのまま終局に
    # 使わず、zobrist hash の出現回数を数えて 4 回目に達したときだけ
    # is_draw() で分類する (cshogi/cli.py と同じ方式)。
    # REPETITION_SUPERIOR / INFERIOR (優等/劣等局面) は探索用ヒューリスティック
    # であり対局の終局ルールではないので無視する。
    seen: Counter[int] = Counter()
    seen[board.zobrist_hash()] += 1
    while True:
        if board.is_game_over():  # 詰み (合法手なし): 手番側の負け
            stm_is_cand = (board.turn == cshogi.BLACK) == cand_is_black
            return (0.0 if stm_is_cand else 1.0), "mate", len(moves)
        if seen[board.zobrist_hash()] >= 4:
            draw = board.is_draw()
            stm_is_cand = (board.turn == cshogi.BLACK) == cand_is_black
            if draw == cshogi.REPETITION_WIN:  # 相手の連続王手による千日手
                return (1.0 if stm_is_cand else 0.0), "perpetual_check_win", len(moves)
            if draw == cshogi.REPETITION_LOSE:  # 自分の連続王手による千日手
                return (0.0 if stm_is_cand else 1.0), "perpetual_check_lose", len(moves)
            return 0.5, "sennichite", len(moves)
        if len(moves) >= max_plies:
            return 0.5, "max_plies", len(moves)

        stm_is_cand = (board.turn == cshogi.BLACK) == cand_is_black
        engine = cand if stm_is_cand else base
        pos = f"position sfen {start_sfen}"
        if moves:
            pos += " moves " + " ".join(moves)
        mv, nodes = engine.bestmove(pos, movetime_ms)
        # 校正監視用 (CPU 負荷混入の検出): 実測 nodes/move を記録する
        if nodes_out is not None and nodes is not None:
            nodes_out["cand" if stm_is_cand else "base"].append(nodes)
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
        seen[board.zobrist_hash()] += 1


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
    ap.add_argument(
        "--fixed-pairs",
        action="store_true",
        help="SPRT 判定で打ち切らず --max-pairs まで対局する (Elo カーブの点推定用)",
    )
    ap.add_argument(
        "--games-dir",
        default=str(HERE / "games"),
        help="出力先ベースディレクトリ (games/<name>/)。他実験から使う場合に指定",
    )
    ap.add_argument("--max-plies", type=int, default=320)
    ap.add_argument("--elo0", type=float, default=-30.0)
    ap.add_argument("--elo1", type=float, default=0.0)
    ap.add_argument("--start-pair", type=int, default=0, help="再開用: このペア番号から始める")
    ap.add_argument(
        "--cand-fv-scale",
        type=int,
        default=None,
        help="候補側 FV_SCALE (未指定=エンジン default 16)",
    )
    ap.add_argument(
        "--base-fv-scale",
        type=int,
        default=None,
        help="基準側 FV_SCALE (未指定=エンジン default 16)",
    )
    args = ap.parse_args()

    out_dir = Path(args.games_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    games_path = out_dir / "games.jsonl"
    summary_path = out_dir / "summary.json"

    with open(args.book) as fh:
        sfens = [line.strip().removeprefix("sfen ").strip() for line in fh if line.strip()]
    if not args.fixed_pairs:
        sfens = sfens[: args.max_pairs]
    # --fixed-pairs では全 sfen を候補に残し、エンジン障害でペアが落ちても
    # 後続の局面で補充して「ちょうど max_pairs ペア」を満たす

    sprt = PentanomialSprt(elo0=args.elo0, elo1=args.elo1)
    # 再開: 既存 games.jsonl から完結ペア (2 局揃い) だけ再構築する。
    # 千切れ行 (前回の hard kill 起因) は捨てる。不完全レコードを落とした場合のみ、
    # tmp ファイル + atomic rename でファイルを圧縮し直す (truncate-in-place だと
    # 書き直し中のクラッシュで全対局記録を失うため)
    done_recs: dict[int, list[dict]] = {}
    if games_path.exists():
        n_lines = 0
        with games_path.open() as fh:
            for line in fh:
                n_lines += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done_recs.setdefault(rec["pair"], []).append(rec)
        done_recs = {p: recs[:2] for p, recs in done_recs.items() if len(recs) >= 2}
        n_kept = sum(len(r) for r in done_recs.values())
        if n_kept != n_lines:
            tmp_path = games_path.with_suffix(".jsonl.tmp")
            with tmp_path.open("w") as fh:
                for p in sorted(done_recs):
                    for rec in done_recs[p]:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, games_path)
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
                fv_scale=args.cand_fv_scale,
            ),
            UsiEngine(
                args.engine,
                args.baseline_evaldir,
                args.threads,
                args.hash_mb,
                args.max_plies,
                stderr_path=out_dir / "engine-base.stderr.log",
                fv_scale=args.base_fv_scale,
            ),
        )

    cand, base = spawn()
    games_fh = games_path.open("a")
    try:
        for i, sfen in enumerate(sfens):
            if args.fixed_pairs and sprt.n_pairs >= args.max_pairs:
                break
            if i < args.start_pair or i in done_recs:
                continue
            recs: list[dict] = []
            # 3 回試行: エンジン silent crash (issue #18) の頻度上昇 (2026-08-02,
            # ~10%/pair) に対し skip 率を p^2 → p^3 に抑える。挙動はリトライ回数のみ
            for attempt in (1, 2, 3):
                try:
                    recs = []
                    for cand_is_black in (True, False):
                        nodes_log: dict[str, list[int]] = {"cand": [], "base": []}
                        score, reason, plies = play_game(
                            cand,
                            base,
                            sfen,
                            cand_is_black,
                            args.movetime,
                            args.max_plies,
                            nodes_out=nodes_log,
                        )
                        med = {
                            k: (sorted(v)[len(v) // 2] if v else None) for k, v in nodes_log.items()
                        }
                        recs.append(
                            {
                                "pair": i,
                                "cand_is_black": cand_is_black,
                                "cand_score": score,
                                "reason": reason,
                                "plies": plies,
                                "sfen": sfen,
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                # 校正監視: 実測 nodes/move の中央値 (CPU 負荷混入検出用)
                                "cand_nodes_med": med["cand"],
                                "base_nodes_med": med["base"],
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
            summary["fv_scale"] = {
                "cand": args.cand_fv_scale or 16,
                "base": args.base_fv_scale or 16,
            }
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
            print(
                f"pair {i + 1}: score={pair_score} | n={summary['pairs']} "
                f"elo={summary['elo']} ci={summary['elo_ci95']} llr={summary['llr']} "
                f"{summary['sprt']}",
                flush=True,
            )
            if summary["sprt"] != "continue" and not args.fixed_pairs:
                print(f"SPRT decision: {summary['sprt']}", flush=True)
                break
    finally:
        cand.quit()
        base.quit()
        games_fh.close()
    print(json.dumps(sprt.summary(), ensure_ascii=False))


if __name__ == "__main__":
    main()
