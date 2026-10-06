"""experiment-011: PC-ALM / PC の checkpoint ができたら、出力側を縮めた nn.bin (export_rescaled.py, k は自動) を作り続ける。

対象: data/bulletou/checkpoints/011-{s,f}-{pcalm,pc}-*/NNNN/ (nn.bin と state.bin があり、--settle 秒以上更新がないもの)。
出力: /mnt/D/sugiyama/011/rescaled/<フォルダ名>-e<N>/nn.bin (+ nn.bin.json)。対局キューのジョブはこちらを指す。

使い方: nohup setsid .venv/bin/python experiments/011-pcalm/rescale_watch.py >> /mnt/D/sugiyama/011/rescale_watch.log 2>&1 &
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CKPT = REPO / "data" / "bulletou" / "checkpoints"
OUT = Path("/mnt/D/sugiyama/011/rescaled")


def ready(d: Path, settle: float) -> bool:
    files = [f for f in d.iterdir() if f.is_file()]
    names = {f.name for f in files}
    return {"nn.bin", "state.bin"} <= names and time.time() - max(
        f.stat().st_mtime for f in files
    ) >= settle


def main() -> None:
    settle = 120.0
    while True:
        for run in sorted(CKPT.glob("011-[sf]-pc*")):
            if not run.is_dir() or ".stopped" in run.name:
                continue
            for ep in sorted(p for p in run.iterdir() if p.is_dir() and p.name.isdigit()):
                dst = OUT / f"{run.name}-e{int(ep.name)}"
                if (dst / "nn.bin.json").exists() or not ready(ep, settle):
                    continue
                print(time.strftime("%F %T"), "export", ep, "->", dst, flush=True)
                r = subprocess.run(
                    [
                        sys.executable,
                        str(HERE / "export_rescaled.py"),
                        "--ckpt",
                        str(ep),
                        "--out",
                        str(dst),
                    ],
                    capture_output=True,
                    text=True,
                )
                print(r.stdout.strip() or r.stderr.strip(), flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
