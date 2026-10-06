"""experiment-011 の対局キュー (experiment-009 の match_queue.py を流用。アブレーションの機能は除いた)。

対局そのものは experiment-009 の match_nodes.py を呼ぶ。エンジンはジョブごとに ``engine`` で指定でき、既定は
experiment-009 と同じバイナリ (~/engines/009-finny-avx2/, suzuki から複製, sha256 423b6b1c…)。


- ジョブファイル (既定 match_queue.toml) はループごとに読み直すので、実行中に追記してよい
- 各ジョブ = (A のネット, B のネット, ノード数, ペア数, 名前)。Elo は A から見た値
  (match_nodes.py の candidate = A, baseline = B)
- ネットの指定:
    "suisho11"                       : ~/suisho11
    "011-s-bp-lr1/0001"              : data/bulletou/checkpoints/ からの相対パス (絶対パスも可)
- 実行できる (入力が揃った) ジョブのうち priority が最小 (同順ならファイル順) のものを実行する。
  揃っていないジョブは待つ (checkpoint は nn.bin があり、ディレクトリ内の全ファイルが
  --settle 秒以上更新されていないとき揃ったとみなす)
- 終了後 notes/ratings.md に 1 行追記して commit + push し、games/<name>/queue_done.json を置く。
  失敗したジョブは games/<name>/queue_failed.json を置いて飛ばす (消せば再試行)
- 再開: match_nodes.py が完結ペア単位で再開する。キューを再起動すれば続きから

使い方 (バックグラウンド):
    nohup setsid nice -n 10 data/matchenv/bin/python experiments/011-pcalm/match_queue.py \
        >> /mnt/D/sugiyama/011/match_queue.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
import tomllib
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = REPO / "data" / "matchenv" / "bin" / "python"
CKPT_ROOT = REPO / "data" / "bulletou" / "checkpoints"
MATCH_NODES = REPO / "experiments" / "009-data-scaling" / "match_nodes.py"
SUISHO11 = Path.home() / "suisho11"
ENGINE = (
    Path.home() / "engines" / "009-finny-avx2" / "YaneuraOu-by-gcc"
)  # experiment-009 と同じ (suzuki から複製)
GAMES = HERE / "games"
RATINGS = HERE / "notes" / "ratings.md"
BRANCH = "exp/011-pcalm"
NO_COMMIT = False

RATINGS_HEADER = """# 対局結果 (レーティング)

`match_queue.py` が対局 1 本ごとに 1 行ずつ追記する (手で書き換えない。備考の追記は可)。

- 条件: 固定ノード (`go nodes N`)、各エンジン Threads 1、hash 256 MB、やねうら王 V9.60 + finny.patch。
  エンジンの sha256 は行ごとに備考に記録する (既定は experiment-009 と同じ `423b6b1c…`)。開始局面は互角局面集 ply24 から
  ペア数に応じた等間隔ストライド (ペア数が同じなら全対局で同じ局面集合)。1 ペア = 先後入替 2 局
- **Elo は A から見た値** (A = `match_nodes.py` の candidate, B = baseline)。95% CI は pentanomial
- pentanomial は A のペア得点 0 / 0.5 / 1 / 1.5 / 2 の件数。勝/分/負は A から見た局数
- **FV_SCALE**: ジョブで `a_fv` / `b_fv` を指定したネットは表記に `@FV<値>` を付ける。無指定は既定の 16
- A / B の表記: `s-bp-lr1-e1` = `011-s-bp-lr1/0001` の checkpoint、`s11` = 水匠 11。sha256 は nn.bin の先頭 12 桁

## 対局キュー (`match_queue.py`)

- ジョブは `match_queue.toml` に `[[job]]` を追記するだけでよい (キューが 1 分ごとに読み直す)。ネットは checkpoint
  (`"011-s-bp-lr1/0001"`) か `"suisho11"`。checkpoint がまだ無いジョブは出来るまで待つ
- 実行できるジョブのうち priority 最小のものから 1 本ずつ、W ワーカー (`nice -n 10`, kajiki は学習中 8 / それ以外 12) で実行する
- 状態: `games/<name>/queue_done.json` (完了)、`queue_failed.json` (失敗。消せば再試行)。
  ログ `/mnt/D/sugiyama/011/match_queue.log`
- 停止: `setsid` でプロセスグループの先頭になっているので、`kill -- -<PID>` でグループごと
  (match_nodes.py とエンジンも) 止める。再起動は下のコマンド (途中の対局は完結ペア単位で再開)
  ```
  nohup setsid nice -n 10 data/matchenv/bin/python experiments/011-pcalm/match_queue.py \\
      >> /mnt/D/sugiyama/011/match_queue.log 2>&1 < /dev/null &
  ```

## 結果

| 日付 | 対局名 | A | B | ノード | ペア | Elo ± 95% CI | pentanomial | 勝/分/負 | 備考 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- nets


def ckpt_dir(spec: str) -> Path:
    p = Path(spec).expanduser()
    return p if p.is_absolute() else CKPT_ROOT / p


def ckpt_ready(d: Path, settle: float) -> bool:
    if not (d / "nn.bin").is_file():
        return False
    newest = max(f.stat().st_mtime for f in d.iterdir() if f.is_file())
    return time.time() - newest >= settle


def net_label(spec) -> str:
    if spec == "suisho11":
        return "s11"
    d = ckpt_dir(spec)
    if d.parent.name.startswith(("011-", "009-")) and d.name.isdigit():
        return f"{d.parent.name[4:]}-e{int(d.name)}"
    return spec


def net_ready(spec, settle: float) -> tuple[bool, str]:
    if spec == "suisho11":
        return True, ""
    d = ckpt_dir(spec)
    return ckpt_ready(d, settle), f"waiting for checkpoint {d}"


def materialize(spec) -> Path:
    """ネットの EvalDir を返す。"""
    return SUISHO11 if spec == "suisho11" else ckpt_dir(spec)


# ---------------------------------------------------------------- jobs


def load_jobs(path: Path, prev: list[dict]) -> list[dict]:
    try:
        jobs = tomllib.loads(path.read_text()).get("job", [])
    except (OSError, tomllib.TOMLDecodeError) as err:
        log(f"cannot read {path}: {err}; keeping previous job list")
        return prev
    for i, j in enumerate(jobs):
        j.setdefault("priority", 100)
        j["_order"] = i
    return sorted(jobs, key=lambda j: (j["priority"], j["_order"]))


def job_state(j: dict) -> str | None:
    d = GAMES / j["name"]
    if (d / "queue_done.json").exists():
        return "done"
    if (d / "queue_failed.json").exists():
        return "failed"
    return None


def engine_of(j: dict) -> Path:
    return Path(j["engine"]).expanduser() if j.get("engine") else ENGINE


def run_match(j: dict, a_dir: Path, b_dir: Path, workers: int) -> dict:
    cmd = [
        str(PY),
        str(MATCH_NODES),
        "--candidate-evaldir", str(a_dir),
        "--baseline-evaldir", str(b_dir),
        "--engine", str(engine_of(j)),
        "--name", j["name"],
        "--games-dir", str(GAMES),
        "--nodes", str(j["nodes"]),
        "--pairs", str(j["pairs"]),
        "--workers", str(j.get("workers", workers)),
    ]  # fmt: skip
    for side, flag in (("a_fv", "--candidate-fv-scale"), ("b_fv", "--baseline-fv-scale")):
        if j.get(side) is not None:
            cmd += [flag, str(j[side])]
    summary = GAMES / j["name"] / "summary.json"
    for attempt in (1, 2, 3):  # 失敗ペアがあれば再実行 (完結ペアは再利用される)
        log(f"match attempt {attempt}: " + shlex.join(cmd))
        subprocess.run(cmd, check=True)
        s = json.loads(summary.read_text())
        if s["pairs"] >= j["pairs"]:
            return s
    raise RuntimeError(f"only {s['pairs']}/{j['pairs']} pairs after 3 attempts")


def wdl(name: str) -> tuple[int, int, int]:
    w = d = lo = 0
    for line in (GAMES / name / "games.jsonl").read_text().splitlines():
        sc = json.loads(line)["cand_score"]
        w += sc == 1.0
        d += sc == 0.5
        lo += sc == 0.0
    return w, d, lo


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def fv_tag(j: dict, side: str) -> str:
    """FV_SCALE を指定したネットは表記に @FV<値> を付ける (無指定 = 既定 16)。"""
    v = j.get(f"{side}_fv")
    return "" if v is None else f"@FV{v}"


def append_rating(j: dict, s: dict, a_dir: Path, b_dir: Path) -> None:
    if not RATINGS.exists():
        RATINGS.write_text(RATINGS_HEADER)
    text = RATINGS.read_text()
    if f"| {j['name']} |" in text:
        log(f"ratings.md already has {j['name']}; not appending")
    else:
        w, d, lo = wdl(j["name"])
        lo_ci, hi_ci = s["elo_ci95"]
        half = (hi_ci - lo_ci) / 2
        notes = [
            f"A {sha256(a_dir / 'nn.bin')[:12]}",
            f"B {sha256(b_dir / 'nn.bin')[:12]}",
            f"engine {sha256(engine_of(j))[:8]}",
        ]
        if s.get("failed_pairs"):
            notes.append(f"失敗ペア {len(s['failed_pairs'])}")
        if j.get("note"):
            notes.append(j["note"])
        row = (
            f"| {time.strftime('%m-%d %H:%M')} | {j['name']} | {net_label(j['a'])}{fv_tag(j, 'a')} | "
            f"{net_label(j['b'])}{fv_tag(j, 'b')} | {j['nodes']:,} | {s['pairs']:,} | "
            f"{s['elo']:+.1f} ± {half:.1f} [{lo_ci:+.1f}, {hi_ci:+.1f}] | "
            f"{'/'.join(map(str, s['pentanomial']))} | {w}/{d}/{lo} | {'; '.join(notes)} |\n"
        )
        with RATINGS.open("a") as fh:
            fh.write(row)
        log("ratings row: " + row.strip())
    commit_push(f"exp: 011 ratings — {j['name']} ({s['elo']:+.1f} Elo, {s['pairs']} pairs)")


def commit_push(title: str) -> None:
    if NO_COMMIT:
        log(f"(test) would commit: {title}")
        return
    br = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if br != BRANCH:
        log(f"on branch {br!r}, not {BRANCH}; ratings.md left uncommitted")
        return
    msg = (
        f"{title}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
        "Claude-Session: https://claude.ai/code/session_01D3WAUtxoJbq57tYmVPuTLS\n"
    )
    rel = str(RATINGS.relative_to(REPO))
    for attempt in range(5):  # index.lock の競合などに備えて再試行
        git("add", rel)
        r = git("commit", "-m", msg, "--", rel)
        if r.returncode == 0 or "nothing to commit" in r.stdout + r.stderr:
            break
        log(f"git commit failed ({attempt}): {r.stderr.strip()}")
        time.sleep(10)
    for attempt in range(3):
        r = git("push", "origin", BRANCH)
        if r.returncode == 0:
            log("pushed")
            return
        log(f"git push failed ({attempt}): {r.stderr.strip()}")
        git("pull", "--rebase", "--autostash", "origin", BRANCH)
        time.sleep(10)
    log("push gave up; commit is local")


def run_job(j: dict, workers: int) -> None:
    log(
        f"=== job {j['name']}: {net_label(j['a'])} vs {net_label(j['b'])}, {j['nodes']} nodes, {j['pairs']} pairs"
    )
    a_dir, b_dir = materialize(j["a"]), materialize(j["b"])
    s = run_match(j, a_dir, b_dir, workers)
    log(f"job {j['name']}: elo {s['elo']} ci {s['elo_ci95']} pairs {s['pairs']}")
    append_rating(j, s, a_dir, b_dir)
    (GAMES / j["name"] / "queue_done.json").write_text(
        json.dumps({"finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "job": j}, ensure_ascii=False)
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jobs", type=Path, default=HERE / "match_queue.toml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--poll", type=float, default=60, help="待ちのときの再確認間隔 (秒)")
    ap.add_argument(
        "--settle", type=float, default=120, help="checkpoint が揃ったとみなす無更新秒数"
    )
    ap.add_argument("--once", action="store_true", help="実行できるジョブが無くなったら終了")
    ap.add_argument(
        "--test-dir",
        type=Path,
        help="テスト用: 棋譜・ratings.md をここに置き、commit しない",
    )
    args = ap.parse_args()
    global GAMES, RATINGS, NO_COMMIT
    if args.test_dir:
        GAMES, RATINGS = args.test_dir / "games", args.test_dir / "ratings.md"
        NO_COMMIT = True

    jobs: list[dict] = []
    last_wait = ""
    while True:
        jobs = load_jobs(args.jobs, jobs)
        names = [j["name"] for j in jobs]
        if len(names) != len(set(names)):
            log("duplicate job names in job file; fix it")
        pending = [j for j in jobs if job_state(j) is None]
        ran = False
        waits = []
        for j in pending:
            ok_a, why_a = net_ready(j["a"], args.settle)
            ok_b, why_b = net_ready(j["b"], args.settle)
            if not (ok_a and ok_b):
                waits.append(f"{j['name']}: {why_a or why_b}")
                continue
            try:
                run_job(j, args.workers)
            except Exception as err:  # noqa: BLE001 — 1 ジョブの失敗でキューを止めない
                log(f"job {j['name']} FAILED: {err}\n{traceback.format_exc()}")
                d = GAMES / j["name"]
                d.mkdir(parents=True, exist_ok=True)
                (d / "queue_failed.json").write_text(json.dumps({"error": str(err)}))
            ran = True
            break
        if ran:
            last_wait = ""
            continue
        if args.once:
            return
        w = "; ".join(waits) if waits else "no pending jobs"
        if w != last_wait:
            log(f"idle: {w}")
            last_wait = w
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
