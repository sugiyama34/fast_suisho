"""USI エンジンドライバ (experiment-003)。

局面リストを 1 局面ずつエンジンに与え、モードに応じて実行する:

- search: `go nodes N` を送り bestmove まで待つ (計測ビルドでの活性化統計収集用。
  結果出力は使わない)
- searchlog: `go nodes N` の最終 info 行 (time/nps を除去) と bestmove を局面ごとに書く。
  Threads=1 の固定ノード探索は決定的なので、オリジナル / permuted の出力 diff が
  「探索中に評価された全局面で評価値が完全一致」の検証になる
  (V9.60 の `eval` コマンドは未実装スタブで静的評価値を直接印字できないため、
  この方式で代替する)

000-baseline の注意点 2 のとおり go 直後に stdin を閉じると探索が中断されるため、
readyok / bestmove を対話的に待つ。

usage:
  usi_drive.py --engine PATH --evaldir DIR --sfens FILE --mode search --nodes 20000
  usi_drive.py --engine PATH --evaldir DIR --sfens FILE --mode searchlog --nodes 50000
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def strip_volatile(info: str) -> str:
    """info 行から実時間依存フィールド (time / nps / hashfull) を除去する。"""
    tokens = info.split()
    out = []
    skip = False
    for tok in tokens:
        if skip:
            skip = False
            continue
        if tok in ("time", "nps", "hashfull"):
            skip = True
            continue
        out.append(tok)
    return " ".join(out)


class Engine:
    def __init__(self, path: str, env: dict[str, str] | None = None):
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
            env=env,
        )

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, prefix: str) -> list[str]:
        """prefix で始まる行が来るまで読み、その行を含む全行を返す。"""
        assert self.proc.stdout is not None
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(f"engine died while waiting for '{prefix}'")
            line = line.rstrip("\n")
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def quit(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--evaldir", required=True)
    ap.add_argument("--sfens", required=True)
    ap.add_argument("--mode", required=True, choices=["search", "searchlog"])
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 局面のみ (0 = 全部)")
    ap.add_argument("--hash-mb", type=int, default=256)
    args = ap.parse_args()

    with open(args.sfens) as f:
        sfens = [line.strip() for line in f if line.strip()]
    if args.limit:
        sfens = sfens[: args.limit]

    eng = Engine(args.engine)
    eng.send(f"setoption name EvalDir value {args.evaldir}")
    eng.send(f"setoption name USI_Hash value {args.hash_mb}")
    eng.send("setoption name Threads value 1")
    eng.send("setoption name USI_OwnBook value false")
    eng.send("setoption name NetworkDelay value 0")
    eng.send("setoption name NetworkDelay2 value 0")
    eng.send("isready")
    eng.read_until("readyok")

    for i, sfen in enumerate(sfens):
        # 行は "sfen ..." 形式なのでそのまま position に渡せる
        eng.send(f"position {sfen}")
        if args.mode == "search":
            eng.send(f"go nodes {args.nodes}")
            eng.read_until("bestmove")
        else:  # searchlog
            eng.send(f"go nodes {args.nodes}")
            lines = eng.read_until("bestmove")
            print(f"=== pos {i} ===")
            infos = [ln for ln in lines if ln.startswith("info ") and " pv " in ln]
            if infos:
                print(strip_volatile(infos[-1]))
            print(lines[-1])
        if (i + 1) % 100 == 0:
            print(f"progress: {i + 1}/{len(sfens)}", file=sys.stderr)

    eng.quit()


if __name__ == "__main__":
    main()
