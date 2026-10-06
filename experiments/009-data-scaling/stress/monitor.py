"""suzuki の負荷試験の記録: GPU の電力・使用率・温度と CPU の使用率・温度を数秒ごとに CSV に書く。

標準ライブラリだけで動く (system の python3)。1 行の要約も標準出力に出す。

    python3 experiments/009-data-scaling/stress/monitor.py [--interval 5]

出力: /mnt/nvme1/sugiyama/stress/monitor-<開始時刻>.csv と、要約行の .txt (`tail -f` 用)。列 `loads` = その時点で動いている負荷
(stress/pids/*.pid の名前, 例 g0 g1 c1)。CPU のパッケージ電力 (RAPL) は root だけが読めるので、
読めなければ空欄 (管理者が `chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj` すると記録される)。
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

STRESS_DIR = Path(os.environ.get("STRESS_DIR", "/mnt/nvme1/sugiyama/stress"))
NGPU = 5
GPU_FIELDS = "index,power.draw,utilization.gpu,temperature.gpu,clocks.sm"
RAPL = [Path(f"/sys/class/powercap/intel-rapl:{i}/energy_uj") for i in (0, 1)]


def gpus() -> list[list[str]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return [["", "", "", ""] for _ in range(NGPU)]
    rows = {}
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        rows[int(f[0])] = f[1:]
    return [rows.get(i, ["", "", "", ""]) for i in range(NGPU)]


def cpu_times() -> tuple[int, int]:
    f = [int(x) for x in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
    idle = f[3] + f[4]  # idle + iowait
    return sum(f[:8]) - idle, sum(f[:8])


def tctl() -> list[str]:
    """k10temp (ソケットごと) の Tctl [°C]。"""
    out = []
    for h in sorted(Path("/sys/class/hwmon").iterdir()):
        try:
            if (h / "name").read_text().strip() == "k10temp":
                out.append(f"{int((h / 'temp1_input').read_text()) / 1000:.1f}")
        except OSError:
            pass
    return out


def cpu_mhz() -> str:
    mhz = [
        float(ln.split(":")[1])
        for ln in Path("/proc/cpuinfo").read_text().splitlines()
        if ln.startswith("cpu MHz")
    ]
    return f"{sum(mhz) / len(mhz):.0f}" if mhz else ""


def rapl() -> list[int] | None:
    try:
        return [int(p.read_text()) for p in RAPL]
    except OSError:
        return None


def procs() -> tuple[int, int]:
    nb = ny = 0
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            comm = (p / "comm").read_text().strip()
        except OSError:
            continue
        nb += comm == "bulletou"
        ny += comm.startswith("YaneuraOu")
    return nb, ny


def loads() -> str:
    """動いている負荷の名前。pid ファイル = 「グループ ID boot_id」(common.sh)。"""
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    names = []
    for f in sorted((STRESS_DIR / "pids").glob("*.pid")):
        try:
            g, b = f.read_text().split()
            if b == boot:
                os.killpg(int(g), 0)
                names.append(f.stem)
        except (OSError, ValueError):
            pass
    return " ".join(names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()
    STRESS_DIR.mkdir(parents=True, exist_ok=True)
    path = STRESS_DIR / f"monitor-{datetime.now():%Y%m%d-%H%M%S}.csv"
    ncpu = os.cpu_count() or 64
    head = ["time"]
    for i in range(NGPU):
        head += [f"gpu{i}_w", f"gpu{i}_util", f"gpu{i}_temp", f"gpu{i}_mhz"]
    head += ["gpu_total_w", "cpu_busy_pct", "cpu_busy_cores", "load1", "cpu_mhz"]
    head += ["tctl0", "tctl1", "cpu_pkg_w", "n_bulletou", "n_yaneuraou", "loads"]
    txt = path.with_suffix(".txt").open("a")
    print(f"writing {path} (+ .txt)", flush=True)
    busy0, total0 = cpu_times()
    e0, t0 = rapl(), time.monotonic()
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        while True:
            time.sleep(args.interval)
            g = gpus()
            busy1, total1 = cpu_times()
            e1, t1 = rapl(), time.monotonic()
            pct = 100.0 * (busy1 - busy0) / max(1, total1 - total0)
            pkg = ""
            if e0 and e1 and all(b >= a for a, b in zip(e0, e1)):
                pkg = f"{sum(b - a for a, b in zip(e0, e1)) / 1e6 / (t1 - t0):.0f}"
            busy0, total0, e0, t0 = busy1, total1, e1, t1
            watts = [float(x[0]) for x in g if x[0] not in ("", "[N/A]")]
            total_w = sum(watts)
            temps = (tctl() + ["", ""])[:2]
            nb, ny = procs()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            load1 = os.getloadavg()[0]
            act = loads()
            row = [now] + [v for x in g for v in x]
            row += [f"{total_w:.0f}", f"{pct:.1f}", f"{pct * ncpu / 100:.1f}", f"{load1:.1f}"]
            row += [cpu_mhz(), *temps, pkg, nb, ny, act]
            w.writerow(row)
            fh.flush()
            gw = "/".join(f"{float(x[0]):.0f}" if x[0] else "-" for x in g)
            gu = "/".join(x[1] or "-" for x in g)
            pk = f" pkg {pkg} W" if pkg else ""
            line = (
                f"{now[11:]}  GPU {gw} W (計 {total_w:.0f} W) util {gu}%  "
                f"CPU {pct:.0f}% ({pct * ncpu / 100:.0f} コア) load {load1:.1f} "
                f"Tctl {'/'.join(temps)}{pk}  [{act}]"
            )
            print(line, flush=True)
            print(line, file=txt, flush=True)


if __name__ == "__main__":
    main()
