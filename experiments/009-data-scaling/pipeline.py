"""experiment-009 の残り全工程を自動で進める supervisor (2026-09-30 ユーザー依頼: 止まらずに自律で最後まで)。

1 分ごとに状態を見て、段階 (stage) を進める。途中で落ちても再起動すれば state.json から続きをやる。
対局は match_queue.py (常駐) に任せ、この script はジョブを match_queue.toml に追記するだけ。

段階:
  full   : full (20 epoch, GPU 0) の学習を見張る (プロセスが消えたら同じコマンドで再開 = 自動 resume)
  pick   : full の FV_SCALE 格子 (e5/e8 は 5 点、e12/e16/e20 は fv_decide.py が追加した 3 点) が揃うのを待ち、
           Elo 最大の (epoch, FV_SCALE) を「full の最良」とする (選択バイアスは許容: ユーザー決定)。
           最終アブレーション (教師順位・対局順位 × 下位 5/10/20/30% × zero/random, 各 1,000 ペア,
           水匠 11@32 と対局, 親の FV_SCALE) と、基準 full の 2,000 ペアを追加
  final  : 上の対局が終わるのを待つ (GPU を使わない時間なので W=56)。loss_eval も流す
  armsA  : データ削減 arm p50/p10/p90/p70/p80 を GPU 0〜4 で E=10 (CPU は軽く: 対局しない)
  armsB  : p60 / p30 を GPU 0〜1 で E=10。並行して armsA の arm の対局 (W=22)。
           終われば残りは W=56
  rate   : 全 arm の対局を待つ → loss_eval → done
arm の対局: 候補 epoch を 300 ペアずつ水匠 11 と対局して最良 epoch を決め、その epoch を 2,000 ペア
(基準 full と同じ開始局面, full の FV_SCALE)。基準は常に full (最良 epoch): arm (最良 epoch) − full と ablated − full を比べて
「X% アブレーション ≈ Y% データ削減」を読む (2026-09-30 ユーザー決定)。

電力 (ユーザー決定): GPU と CPU の併用は GPU 3 枚 + CPU 32 コアまで。CPU が軽ければ GPU 5 枚、GPU を使わなければ CPU 全部。

使い方: nohup setsid data/matchenv/bin/python experiments/009-data-scaling/pipeline.py \
            >> /mnt/nvme1/sugiyama/logs/pipeline.log 2>&1 < /dev/null &
        --dry-run: 1 回だけ状態を判定して表示し、何も起動・追記しない
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = REPO / "data" / "matchenv" / "bin" / "python"
TRAIN_PY = REPO / ".venv" / "bin" / "python"
CKPT = REPO / "data" / "bulletou" / "checkpoints"
GAMES = HERE / "games"
TOML = HERE / "match_queue.toml"
STATUS_MD = HERE / "notes" / "pipeline-status.md"
ROOT = Path("/mnt/nvme1/sugiyama")
STATE = ROOT / "pipeline" / "state.json"
LOGS = ROOT / "logs"
FV_FLAG = ROOT / "fv" / "full.decided"
SEARCH = ROOT / "feature_counts" / "search_s11_2000_300k.npy"
ABLATED = ROOT / "ablated"
LOSS_ROOT = ROOT / "loss_eval"
CUDA_HOME = ROOT / "cuda-12.8"
BRANCH = "exp/009-data-scaling"

FULL_EPOCHS = 20
ABL_PAIRS = 667  # 10-01: 1,000 → 667 (開始局面は full の 2,000 ペアの部分集合のまま。締切のため)
ABL_PCTS = (5, 10, 20, 30, 50, 75)  # 10-01 ユーザー: 30% まで効果がほぼ無いので 50/75% を追加
ARM_EPOCHS = 10
ARMS_A = {"p50": "0", "p10": "1", "p90": "2", "p70": "3", "p80": "4"}  # arm -> GPU
# 2 波目: p60 と p30 (p30 は 50%→90% 削減の間を埋める)。GPU 2 枚 + 対局 (電力の予算内)
ARMS_B = {"p60": "0", "p30": "1"}
FRACTION = {
    "p90": 0.9,
    "p80": 0.8,
    "p70": 0.7,
    "p60": 0.6,
    "p50": 0.5,
    "p30": 0.3,
    "p10": 0.1,
    "full-rot": 1.0,
}
ARM_PAIRS = 2000  # 基準 full と arm の本命対局 (データ側は棋譜が全部変わるので対局数を増やす)
W_ARMS_B = 22  # 2 波目 (GPU 3 枚, 学習プロセスが計約 9 コア) と並行する対局の並列数
GRID5 = (32, 40, 48, 56, 64)
S11 = 'b = "suisho11"\nb_fv = 32'
DRY = False


def log(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def running(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


# ---------------------------------------------------------------- state


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"stage": "full", "restarts": {}, "history": []}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, ensure_ascii=False))
    os.replace(tmp, STATE)


def advance(st: dict, stage: str, note: str) -> None:
    log(f"stage {st['stage']} -> {stage}: {note}")
    st["history"].append(
        {"time": time.strftime("%F %T"), "from": st["stage"], "to": stage, "note": note}
    )
    st["stage"] = stage
    save_state(st)
    write_status(st)


# ---------------------------------------------------------------- training


def ckpt_has(arm: str, ep: int) -> bool:
    return (CKPT / f"009-{arm}" / f"{ep:04d}" / "nn.bin").is_file()


def training_alive(arm: str) -> bool:
    return running(f"train_supervised.py --arm {arm} --gpu")


def gpu_ok() -> bool:
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def ensure_training(st: dict, arm: str, gpu: str, epochs: int) -> bool:
    """学習が終わっていれば True。動いていなければ起動 (既存 checkpoint から自動 resume)。"""
    if ckpt_has(arm, epochs) and not training_alive(arm):
        return True
    if training_alive(arm):
        return False
    n = st["restarts"].get(arm, 0)
    if n >= 6:
        log(f"{arm}: restarted {n} times already; not restarting (needs a human)")
        return False
    if not gpu_ok():
        log(f"{arm}: nvidia-smi fails; waiting")
        return False
    log(f"launch training {arm} on GPU {gpu}, {epochs} epochs (launch #{n + 1})")
    if DRY:
        return False
    env = {**os.environ, "CUDA_HOME": str(CUDA_HOME)}
    with (LOGS / f"train-{arm}.log").open("a") as fh:
        subprocess.Popen(
            [
                str(TRAIN_PY),
                str(HERE / "train_supervised.py"),
                "--arm",
                arm,
                "--gpu",
                gpu,
                "--epochs",
                str(epochs),
            ],
            cwd=REPO,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    st["restarts"][arm] = n + 1
    save_state(st)
    return False


# ---------------------------------------------------------------- queue


def ensure_daemons() -> None:
    if not running("match_queue.py --workers"):
        log("match_queue is not running; starting it (W=24)")
        if not DRY:
            with (LOGS / "match_queue.log").open("a") as fh:
                subprocess.Popen(
                    ["nice", "-n", "10", str(PY), str(HERE / "match_queue.py"), "--workers", "24"],
                    cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )  # fmt: skip
    if not FV_FLAG.exists() and not running("fv_decide.py"):
        log("fv_decide.py is not running and FV_SCALE is undecided; starting it")
        if not DRY:
            with (LOGS / "fv_decide.log").open("a") as fh:
                subprocess.Popen(
                    [str(PY), str(HERE / "fv_decide.py")], cwd=REPO, stdout=fh,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                )  # fmt: skip


def job_done(name: str) -> bool:
    return (GAMES / name / "queue_done.json").exists()


def job_failed(name: str) -> bool:
    return (GAMES / name / "queue_failed.json").exists()


def job_elo(name: str) -> float:
    return json.loads((GAMES / name / "summary.json").read_text())["elo"]


def queue_busy() -> bool:
    return running("match_nodes.py")


def append_jobs(marker: str, text: str) -> None:
    """match_queue.toml に追記 (marker が既にあれば何もしない)。一時ファイル経由で置き換える。"""
    s = TOML.read_text()
    if marker in s:
        return
    log(f"append jobs: {marker}")
    if DRY:
        return
    tmp = TOML.with_suffix(".toml.pipeline.tmp")
    tmp.write_text(s + "\n" + marker + "\n" + text)
    os.replace(tmp, TOML)


def set_workers(prefix: str, workers: int) -> None:
    """未実行のジョブ (名前が prefix で始まる) の workers を書き換える。"""
    s = TOML.read_text()
    blocks = re.split(r"(?m)^\[\[job\]\]\n", s)
    out, changed = [blocks[0]], 0
    for b in blocks[1:]:
        m = re.search(r'(?m)^name = "([^"]+)"', b)
        if (
            m
            and m.group(1).startswith(prefix)
            and not job_done(m.group(1))
            and not running(m.group(1))
        ):
            b2 = re.sub(r"(?m)^workers = \d+$", f"workers = {workers}", b)
            changed += b2 != b
            b = b2
        out.append("[[job]]\n" + b)
    if changed and not DRY:
        tmp = TOML.with_suffix(".toml.pipeline.tmp")
        tmp.write_text("".join(out))
        os.replace(tmp, TOML)
        log(f"set workers={workers} for {changed} pending jobs '{prefix}*'")


def job(
    name: str,
    a: str,
    b: str,
    pairs: int,
    priority: int,
    workers: int,
    note: str,
    a_fv: int | None = None,
) -> str:
    fv = "" if a_fv is None else f"a_fv = {a_fv}\n"
    return (
        f'[[job]]\nname = "{name}"\na = {a}\n{fv}{b}\nnodes = 300000\npairs = {pairs}\n'
        f'priority = {priority}\nworkers = {workers}\nnote = "{note}"\n\n'
    )


# ---------------------------------------------------------------- loss


def loss_eval(net: Path, label: str) -> Path:
    out = LOSS_ROOT / label
    if not (out / "loss.json").exists() and not DRY:
        log(f"loss_eval {label}")
        subprocess.run(
            [str(PY), str(HERE / "loss_eval" / "loss_eval.py"), "--net", str(net), "--label", label, "--threads", "8"],
            cwd=REPO, check=False, stdout=subprocess.DEVNULL,
        )  # fmt: skip
    return out


def loss_diff(parent: Path, child: Path, label: str) -> None:
    out = LOSS_ROOT / f"diff-{label}.json"
    if (
        out.exists()
        or DRY
        or not (parent / "loss.json").exists()
        or not (child / "loss.json").exists()
    ):
        return
    subprocess.run(
        [str(PY), str(HERE / "loss_eval" / "loss_eval.py"), "--diff", str(parent), str(child), "--out", str(out)],
        cwd=REPO, check=False, stdout=subprocess.DEVNULL,
    )  # fmt: skip


# ---------------------------------------------------------------- job lists


def full_grid_names(st: dict) -> list[str]:
    names = [f"fv-full-e{ep}@{fv}-vs-s11@32-300k" for ep in (5, 8) for fv in GRID5]
    f = int(FV_FLAG.read_text().strip())
    names += [
        f"fv-full-e{ep}@{fv}-vs-s11@32-300k" for ep in (12, 16, 20) for fv in (f - 8, f, f + 8)
    ]
    # 10-01 追加: e12/e16 で上端 (f + 8 = 48) が最良だったので、e16/e20 は f + 16 (= 56) も測る
    names += [f"fv-full-e{ep}@{f + 16}-vs-s11@32-300k" for ep in (16, 20)]
    return names


def final_jobs(ep: int, fv: int) -> tuple[list[str], str]:
    names, text = [], ""
    for mode, m, pr in (("zero-specific", "zs", 5), ("random-specific", "rs", 6)):
        for rank in ("teach", "match"):
            for pct in ABL_PCTS:
                counts = "--arm full" if rank == "teach" else f"--counts {SEARCH}"
                sel = f"{counts} --percent {pct} --by features --report-counts {SEARCH}"
                abl = f"full-e{ep}-{m}-{rank}-f{pct}"
                a = f'{{ ablate = "{abl}", ckpt = "009-full/{ep:04d}", select = "{sel}", mode = "{mode}" }}'
                name = f"final-{abl}@{fv}-vs-s11@32-300k"
                names.append(name)
                text += job(
                    name,
                    a,
                    S11,
                    ABL_PAIRS,
                    pr,
                    56,
                    "最終アブレーション (親と同じ FV_SCALE, 水匠 11 と対局)",
                    a_fv=fv,
                )
    # 基準は常に full (最良 epoch・最良 FV_SCALE, ユーザー決定)。2,000 ペアで水匠 11 と対局する。
    # アブレーション (1,000 ペア) と arm (2,000 ペア) の開始局面はこの部分集合 / 同一なので、局面単位で対応が取れる
    name = f"final-full-e{ep}@{fv}-vs-s11@32-300k-{ARM_PAIRS}p"
    names.append(name)
    text += job(
        name,
        f'"009-full/{ep:04d}"',
        S11,
        ARM_PAIRS,
        4,
        56,
        "基準 full (アブレーション・データ削減の共通の基準)",
        a_fv=fv,
    )
    return names, text


def eq_epoch(arm: str) -> int:
    return max(1, round(ARM_EPOCHS * FRACTION[arm]))


SEL_PAIRS = 300  # 最良 epoch を選ぶための対局 (選択バイアスは許容: ユーザー決定)


def sel_epochs(arm: str) -> tuple[int, ...]:
    """最良 epoch の候補。1 epoch = 43.2 億局面なので p10 は 1 epoch で約 2.9 周 (最良は 4〜5 周付近と予想)。"""
    return (1, 2, 3, 4, 6) if arm == "p10" else (2, 4, 6, 8, 10)


def arm_jobs(arm: str, fv: int, workers: int) -> tuple[list[str], str]:
    """arm の対局 (水匠 11 相手, full の FV_SCALE)。

    - データ削減 arm: 候補 epoch を 300 ペアずつ → 最良 epoch を決めて 2,000 ペア (arm_followups が追加)
    - full-rot (複製): e10 を 2,000 ペア (full-e10 と比べて学習の揺らぎを見る)
    """
    names, text = [], ""
    if arm == "full-rot":
        name = f"arm-full-rot-e10@{fv}-vs-s11@32-300k-{ARM_PAIRS}p"
        names.append(name)
        text += job(
            name,
            '"009-full-rot/0010"',
            S11,
            ARM_PAIRS,
            12,
            workers,
            "学習の揺らぎ (full-e10 と比べる)",
            a_fv=fv,
        )
        return names, text
    for e in sel_epochs(arm):
        name = f"arm-{arm}-sel-e{e}@{fv}-vs-s11@32-300k"
        names.append(name)
        text += job(
            name,
            f'"009-{arm}/{e:04d}"',
            S11,
            SEL_PAIRS,
            10,
            workers,
            "arm の最良 epoch の選択",
            a_fv=fv,
        )
    return names, text


def arm_followups(st: dict, arms: list[str], workers: int) -> None:
    """候補 epoch の対局が揃った arm について、最良 epoch の本命対局 (基準 full と同じ 2,000 ペア) を追加する。"""
    fv = st["best"]["fv"]
    st.setdefault("arm_best", {})
    for arm in arms:
        if arm == "full-rot" or arm in st["arm_best"]:
            continue
        sel = [f"arm-{arm}-sel-e{e}@{fv}-vs-s11@32-300k" for e in sel_epochs(arm)]
        if not all(job_done(n) or job_failed(n) for n in sel):
            continue
        elos = {n: job_elo(n) for n in sel if job_done(n)}
        if not elos:
            continue
        e = int(re.search(r"-sel-e(\d+)@", max(elos, key=elos.get)).group(1))
        name = f"arm-{arm}-best-e{e}@{fv}-vs-s11@32-300k-{ARM_PAIRS}p"
        text = job(
            name,
            f'"009-{arm}/{e:04d}"',
            S11,
            ARM_PAIRS,
            12,
            workers,
            f"{arm} の最良 epoch (基準 full と同じ 2,000 ペア)",
            a_fv=fv,
        )
        append_jobs(f"# pipeline.py が追加: {arm} の最良 epoch = e{e}", text)
        st["arm_best"][arm] = {"epoch": e, "job": name, "sel": elos}
        st["arm_jobs"] = st["arm_jobs"] + [name]
        save_state(st)
        log(f"{arm}: best epoch e{e} ({elos})")


# ---------------------------------------------------------------- status note


def write_status(st: dict) -> None:
    lines = [
        "# pipeline の進捗 (自動生成)\n",
        "`pipeline.py` が段階を進めるたびに書き換える (手で編集しない)。対局の結果は `ratings.md`。\n",
        f"- 現在の段階: **{st['stage']}** ({time.strftime('%F %T')} 更新)",
    ]
    if "best" in st:
        b = st["best"]
        lines.append(
            f"- full の最良: e{b['epoch']} @ FV_SCALE {b['fv']} (水匠 11@32 に対し {b['elo']:+.1f} Elo)"
        )
    lines += ["\n| 時刻 | 段階 | 内容 |", "| --- | --- | --- |"]
    lines += [f"| {h['time']} | {h['from']} → {h['to']} | {h['note']} |" for h in st["history"]]
    if DRY:
        print("\n".join(lines))
        return
    STATUS_MD.write_text("\n".join(lines) + "\n")
    commit_push([STATUS_MD], f"exp: 009 pipeline — stage {st['stage']}")


def commit_push(paths: list[Path], title: str) -> None:
    """paths だけを commit して push する (対局キューの commit と競合したら再試行)。"""
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)  # noqa: E731
    if git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() != BRANCH:
        return
    rels = [str(p.relative_to(REPO)) for p in paths]
    msg = (
        f"{title}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
        "Claude-Session: https://claude.ai/code/session_01YJdawcYFSvVniYuyJpM5c4\n"
    )
    for _ in range(5):
        git("add", "--", *rels)
        r = git("commit", "-m", msg, "--", *rels)
        if r.returncode == 0 or "nothing to commit" in r.stdout + r.stderr:
            break
        time.sleep(10)
    for _ in range(3):
        if git("push", "origin", BRANCH).returncode == 0:
            return
        git("pull", "--rebase", "--autostash", "origin", BRANCH)
        time.sleep(10)


def refresh_results(st: dict) -> None:
    """15 分ごとに図と notes/results.md を作り直し、変わっていれば push する。"""
    if DRY or time.time() - st.get("last_plot", 0) < 900:
        return
    st["last_plot"] = time.time()
    save_state(st)
    r = subprocess.run(
        [str(TRAIN_PY), str(HERE / "plot_results.py")], cwd=REPO, capture_output=True, text=True
    )
    if r.returncode != 0:
        log(f"plot_results failed: {r.stderr.strip()[-300:]}")
        return
    note = HERE / "notes" / "results.md"
    figs = sorted((HERE / "figures").glob("*.png"))
    status = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--", str(note), str(HERE / "figures")],
        capture_output=True, text=True,
    ).stdout  # fmt: skip
    if status.strip():
        log("results changed; commit figures and results.md")
        commit_push([note, *figs], "exp: 009 results — figures and results.md (auto)")


# ---------------------------------------------------------------- main loop


def step(st: dict) -> None:
    ensure_daemons()
    refresh_results(st)
    stage = st["stage"]

    if stage == "full":
        if ensure_training(st, "full", "0", FULL_EPOCHS):
            advance(st, "pick", "full の学習 (20 epoch) が終了")

    elif stage == "pick":
        if not FV_FLAG.exists():
            return
        names = full_grid_names(st)
        missing = [n for n in names if not job_done(n)]
        if missing:
            if all(job_failed(n) or job_done(n) for n in names):
                log(
                    f"grid jobs failed: {[n for n in names if job_failed(n)]}; using the finished ones"
                )
            else:
                return
        elos = {n: job_elo(n) for n in names if job_done(n)}
        best_name = max(elos, key=elos.get)
        m = re.match(r"fv-full-e(\d+)@(\d+)-", best_name)
        ep, fv = int(m.group(1)), int(m.group(2))
        st["best"] = {"epoch": ep, "fv": fv, "elo": elos[best_name], "grid": elos}
        fnames, text = final_jobs(ep, fv)
        st["final_jobs"] = fnames
        append_jobs(
            f"# 8. pipeline.py が追加: full 最良 e{ep} @ FV_SCALE {fv} の最終アブレーション等", text
        )
        advance(
            st,
            "final",
            f"full の最良 = e{ep} @ FV_SCALE {fv} ({elos[best_name]:+.1f} Elo)。最終アブレーションを追加",
        )

    elif stage == "final":
        ep, fv = st["best"]["epoch"], st["best"]["fv"]
        parent = loss_eval(CKPT / "009-full" / f"{ep:04d}" / "nn.bin", f"009-full-{ep:04d}")
        for d in sorted(ABLATED.glob(f"full-e{ep}-*")):
            if (d / "nn.bin.json").is_file():
                child = loss_eval(d / "nn.bin", d.name)
                loss_diff(parent, child, d.name)
        names = st["final_jobs"]
        if all(job_done(n) or job_failed(n) for n in names):
            if queue_busy():
                return
            advance(
                st,
                "armsA",
                "最終アブレーションの対局が終了。arm p50/p10/p90/p70/p80 を GPU 5 枚で学習開始",
            )

    elif stage == "armsA":
        if queue_busy():  # GPU 5 枚のときは CPU を軽く (対局が残っていれば終わるまで起動しない)
            return
        done = [ensure_training(st, a, g, ARM_EPOCHS) for a, g in ARMS_A.items()]
        if all(done):
            fv = st["best"]["fv"]
            names, text = [], ""
            for a in ARMS_A:
                n, t = arm_jobs(a, fv, W_ARMS_B)
                names += n
                text += t
            st["arm_jobs"] = names
            append_jobs("# 9. pipeline.py が追加: arm (p50/p10/p90/p70/p80) の対局", text)
            advance(
                st,
                "armsB",
                f"arm 5 本の学習が終了。p60/p30 を学習開始、並行して arm の対局 (W={W_ARMS_B})",
            )

    elif stage == "armsB":
        arm_followups(st, list(ARMS_A), W_ARMS_B)
        done = [ensure_training(st, a, g, ARM_EPOCHS) for a, g in ARMS_B.items()]
        if all(done):
            fv = st["best"]["fv"]
            names, text = [], ""
            for a in ARMS_B:
                n, t = arm_jobs(a, fv, 56)
                names += n
                text += t
            st["arm_jobs"] = st["arm_jobs"] + names
            append_jobs("# 10. pipeline.py が追加: arm p60/p30 の対局", text)
            set_workers("arm-", 56)
            advance(st, "rate", "2 波目 (p60/p30) の学習が終了。残りの対局は W=56")

    elif stage == "rate":
        arms = list(ARMS_A) + list(ARMS_B)
        arm_followups(st, arms, 56)
        for a, info in st.get("arm_best", {}).items():
            e = info["epoch"]
            loss_eval(CKPT / f"009-{a}" / f"{e:04d}" / "nn.bin", f"009-{a}-{e:04d}")
        data_arms = [a for a in arms if a != "full-rot"]
        if all(a in st.get("arm_best", {}) for a in data_arms) and all(
            job_done(n) or job_failed(n) for n in st["arm_jobs"]
        ):
            advance(st, "done", "全工程が終了")


def main() -> None:
    global DRY
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll", type=float, default=60)
    args = ap.parse_args()
    DRY = args.dry_run
    st = load_state()
    log(f"pipeline start (stage {st['stage']}, dry={DRY})")
    if DRY:
        step(st)
        write_status(st)
        return
    if not STATUS_MD.exists():
        write_status(st)
    while st["stage"] != "done":
        try:
            step(st)
        except Exception as err:  # 1 回の失敗で supervisor を止めない
            log(f"step error: {err!r}")
        time.sleep(args.poll)
    log("pipeline done")


if __name__ == "__main__":
    main()
