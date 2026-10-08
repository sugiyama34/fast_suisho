"""experiment-011 M4: 本番規模の各手法の checkpoint × FV_SCALE の選択対局と、最終対局を自動で積む (常駐)。

experiment-009 の full の手順 (fv_decide.py / pipeline.py) を両手法に同じ規則で当てる (hypothesis.md §4.4):

- e5, e8: FV_SCALE 32, 40, 48, 56, 64 (各 400 ペア, 水匠 11 @ 32 と対局)
- f = e8 の格子で Elo 最大の FV_SCALE。e12: f−8, f, f+8。e16, e20: f−8, f, f+8, f+16
- **端の規則** (両手法に共通): ある epoch の格子で端の値が最良なら、その外側に 1 点ずつ足す (内側になるまで)
- 全点 (400 ペア) の Elo 最大を最良とし、2,000 ペアで測り直す (水匠 11 @ 32)
- 両手法の最良が決まったら、直接対局 (PC-ALM の最良 vs BP の最良, 2,000 ペア, 同じ開始局面)

PC 系のネットは出力側を 1/k (k = 2) に縮めた nn.bin (rescale_watch.py が作る) を使い、FV_SCALE も 1/k
(32〜64 → 16〜32、刻み 8 → 4) にする。BP は k = 1 (元の nn.bin)。エンジンは match_queue.py の既定
(experiment-009 と同じバイナリ)。ジョブは match_queue.toml に追記するだけで、対局は match_queue.py が行う。

状態: /mnt/D/sugiyama/011/select_state.json (表示用)。何度再起動してもよい (toml と games/ から毎回判定する)。

使い方: nohup setsid .venv/bin/python experiments/011-pcalm/select_queue.py >> /mnt/D/sugiyama/011/select_queue.log 2>&1 &
        --dry-run: 1 回判定して積むはずのジョブを表示するだけ
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOML = HERE / "match_queue.toml"
GAMES = HERE / "games"
STATE = Path("/mnt/D/sugiyama/011/select_state.json")
RESCALED = Path("/mnt/D/sugiyama/011/rescaled")

CKPT = HERE.parents[1] / "data" / "bulletou" / "checkpoints"
RUNS = {
    "bp": {"arm": "f-bp-lr0.7-lrmin3", "k": 1, "label": "full-bp-lr0.7-lrmin3"},
    "pcalm": {"arm": "f-pcalm-T4-gn", "k": 2, "label": "full-pcalm-T4-gn"},
}
BASE = (32, 40, 48, 56, 64)
EPOCHS = (5, 8, 12, 16, 20)
SEL_PAIRS = 400
FINAL_PAIRS = 2000
OPP = 'b = "suisho11"\nb_fv = 32'


def runs() -> dict:
    """主の 2 手法に、本番規模の PC (α=0) の run があれば加える (T は start-full-pc.sh が調整対局で決める)。
    PC は格子と 2,000 ペアの測り直しまで (副次の結果, hypothesis.md §4.5)。直接対局は PC-ALM vs BP のみ。"""
    out = dict(RUNS)
    pcs = sorted(d.name for d in CKPT.glob("011-f-pc-T*-gn") if d.is_dir())
    if len(pcs) == 1:
        arm = pcs[0].removeprefix("011-")
        out["pc"] = {"arm": arm, "k": 2, "label": "full-" + arm.removeprefix("f-")}
    return out


def net(run: dict, ep: int) -> str:
    if run["k"] == 1:
        return f'"011-{run["arm"]}/{ep:04d}"'
    return f'"{RESCALED}/011-{run["arm"]}-e{ep}"'


def sel_name(run: dict, ep: int, fv: int) -> str:
    return f"sel-{run['label']}-e{ep}@{fv}-vs-s11@32-300k"


def existing_jobs() -> set[str]:
    return set(re.findall(r'(?m)^name = "([^"]+)"', TOML.read_text()))


def result(name: str) -> float | None:
    d = GAMES / name
    if not (d / "queue_done.json").exists():
        return None
    return float(json.loads((d / "summary.json").read_text())["elo"])


def failed(name: str) -> bool:
    return (GAMES / name / "queue_failed.json").exists()


def job_text(name: str, a: str, a_fv: int, b: str, pairs: int, priority: int, note: str) -> str:
    return (
        f'[[job]]\nname = "{name}"\na = {a}\na_fv = {a_fv}\n{b}\nnodes = 300000\npairs = {pairs}\n'
        f'priority = {priority}\nnote = "{note}"\n\n'
    )


def plan_run(key: str, run: dict, have: set[str]) -> tuple[list[str], dict]:
    """このラウンドで積むジョブ (TOML 片) と状態を返す。"""
    k = run["k"]
    g = 8 // k
    base = [x // k for x in BASE]
    out: list[str] = []
    state: dict = {"epochs": {}}

    def want(ep: int, fv: int, pr: int) -> None:
        name = sel_name(run, ep, fv)
        if name not in have:
            note = (
                f"M4 選択 ({key}, e{ep}, FV_SCALE {fv}"
                + (f" = {fv * k} 相当, 出力側 1/{k}" if k != 1 else "")
                + ")"
            )
            out.append(job_text(name, net(run, ep), fv, OPP, SEL_PAIRS, pr, note))
            have.add(name)

    def grid_of(ep: int) -> list[int]:
        pat = re.compile(rf"^sel-{re.escape(run['label'])}-e{ep}@(\d+)-vs-s11@32-300k$")
        return sorted(int(m.group(1)) for n in have if (m := pat.match(n)))

    def extend(ep: int, pr: int) -> tuple[bool, int | None]:
        """格子が全部終わっていれば端の規則を当てる。戻り値 (この epoch が確定したか, 最良の FV)。"""
        fvs = grid_of(ep)
        res = {fv: result(sel_name(run, ep, fv)) for fv in fvs}
        state["epochs"][ep] = {fv: r for fv, r in res.items()}
        # 失敗したジョブがあると格子が揃わず、最終対局が積まれないまま止まる → 状態に残してログに出す (手で直す)
        state.setdefault("failed", []).extend(
            sel_name(run, ep, fv) for fv in fvs if failed(sel_name(run, ep, fv))
        )
        if not fvs or any(r is None for r in res.values()):
            return False, None
        best = max(res, key=res.get)
        if best == fvs[0] and best - g > 0:
            want(ep, best - g, pr)
            return False, None
        if best == fvs[-1]:
            want(ep, best + g, pr)
            return False, None
        return True, best

    for ep in (5, 8):
        for fv in base:
            want(ep, fv, 30)
    done5, _ = extend(5, 30)
    done8, f = extend(8, 30)
    state["f"] = f
    complete = done5 and done8
    if f is not None:
        for ep, pts in (
            (12, (f - g, f, f + g)),
            (16, (f - g, f, f + g, f + 2 * g)),
            (20, (f - g, f, f + g, f + 2 * g)),
        ):
            for fv in pts:
                want(ep, fv, 30 + EPOCHS.index(ep))
            d, _ = extend(ep, 30 + EPOCHS.index(ep))
            complete = complete and d
    else:
        complete = False
    if complete:
        allres = {
            (ep, fv): r
            for ep in EPOCHS
            for fv, r in state["epochs"].get(ep, {}).items()
            if r is not None
        }
        (bep, bfv), belo = max(allres.items(), key=lambda kv: kv[1])
        state["best"] = {"epoch": bep, "fv": bfv, "elo400": belo}
        name = f"final-{run['label']}-e{bep}@{bfv}-vs-s11@32-300k-{FINAL_PAIRS}p"
        state["final"] = {"name": name, "elo": result(name)}
        if name not in have:
            note = f"M4 最終 ({key} の最良, 2,000 ペア)" + (
                f", FV_SCALE {bfv} = {bfv * k} 相当" if k != 1 else ""
            )
            out.append(job_text(name, net(run, bep), bfv, OPP, FINAL_PAIRS, 40, note))
            have.add(name)
    return out, state


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll", type=float, default=120.0)
    args = ap.parse_args()
    reported: list[str] = []
    while True:
        have = existing_jobs()
        add: list[str] = []
        states = {}
        for key, run in runs().items():
            jobs, st = plan_run(key, run, have)
            add += jobs
            states[key] = st
        if all("best" in states[k] for k in ("bp", "pcalm")):
            pb, bb = states["pcalm"]["best"], states["bp"]["best"]
            name = (
                f"direct-{RUNS['pcalm']['label']}-e{pb['epoch']}@{pb['fv']}"
                f"-vs-{RUNS['bp']['label']}-e{bb['epoch']}@{bb['fv']}-300k-{FINAL_PAIRS}p"
            )
            states["direct"] = {"name": name, "elo": result(name)}
            if name not in have:
                b = f"b = {net(RUNS['bp'], bb['epoch'])}\nb_fv = {bb['fv']}"
                add.append(
                    job_text(
                        name,
                        net(RUNS["pcalm"], pb["epoch"]),
                        pb["fv"],
                        b,
                        FINAL_PAIRS,
                        45,
                        "M4 直接対局 (PC-ALM の最良 vs BP の最良, 同じ開始局面)",
                    )
                )
        bad = sorted(
            n for st in states.values() if isinstance(st, dict) for n in st.get("failed", [])
        )
        if bad != reported:
            print(
                time.strftime("%F %T"),
                "!!! FAILED selection job(s) — grid is stuck until fixed:",
                bad,
                flush=True,
            )
            reported = bad
        if args.dry_run:
            print("".join(add) or "(nothing to add)")
            print(json.dumps(states, indent=1, default=str))
            return
        if add:
            text = TOML.read_text()
            if not text.endswith("\n"):
                text += "\n"
            text += "\n# M4 (select_queue.py が自動で追加)\n" + "".join(add)
            tmp = TOML.with_suffix(".toml.tmp")
            tmp.write_text(text)
            os.replace(tmp, TOML)
            print(
                time.strftime("%F %T"),
                f"added {len(add)} job(s):",
                [re.search(r'name = "([^"]+)"', a).group(1) for a in add],
                flush=True,
            )
        STATE.write_text(
            json.dumps({"updated": time.strftime("%F %T"), **states}, indent=1, default=str)
        )
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
