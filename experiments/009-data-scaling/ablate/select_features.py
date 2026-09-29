"""出現回数の配列から「稀な特徴量」の一覧 (.npy) を作る (experiment-009 のアブレーション用)。

対象 (分母) は構造的に現れ得る特徴量 (feature_count/analyze.py の structural_impossible() の補集合,
123,053 個) だけ。構造的に現れ得ない特徴量は絶対に選ばない。

選び方 (--by):
  features    : 出現回数の少ない順に、現れ得る特徴量の X% (個数 = round(X/100 × 123,053)) を選ぶ
  occurrences : 出現回数の少ない順に、累積出現回数が総出現回数の X% 以下に収まる最大の集合を選ぶ

同数 (タイ) の順序 (--tie-break):
  random (既定) : 出現回数が同じ特徴量は、--seed で固定した乱数順列で並べる (決定的)。
                  index 順だと境界のタイが玉の筋の若い側に偏るのを避けるため
  index         : 特徴量 index の昇順
境界でタイが割られた場合 (同じ回数の特徴量の一部だけが選ばれた場合) は JSON に件数を記録する。

カウントの指定:
  --arm p10                 : subsets.py の arm のファイル (dlsuisho_unique_NNN.<kind>.npy) の和
  --ckpt <dir>              : checkpoint の teacher.txt に書かれたファイルの和
  --counts a.npy b.npy ...  : 任意の配列の和 (対局側のカウント等)
  --kind both (既定) | dropped_both | stm | nstm。both = loss 重み > 0 のレコードだけ

使い方:
  data/matchenv/bin/python experiments/009-data-scaling/ablate/select_features.py \\
      --arm p10 --percent 5 --out rare_p10_5pct.npy
  (<out>.json に選択数・境界の出現回数・タイ・出現回数のシェアを書く。--report-counts で
   別の配列 (例: 対局側) に対するシェアも記録)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "feature_count"))
sys.path.insert(0, str(HERE.parent))
from analyze import structural_impossible  # noqa: E402

COUNT_DIR = Path("/mnt/nvme1/sugiyama/feature_counts")


def sum_counts(paths: list[Path]) -> np.ndarray:
    return sum(np.load(p).astype(np.uint64) for p in paths)


def count_paths(args: argparse.Namespace) -> list[Path]:
    if args.counts:
        return args.counts
    if args.arm:
        from subsets import ARMS

        nums = ARMS[args.arm]
    else:
        text = (args.ckpt / "teacher.txt").read_text()
        nums = [int(m) for m in re.findall(r"dlsuisho_unique_(\d{3})\.psv", text)]
        if not nums:
            raise SystemExit(f"no dlsuisho_unique_NNN.psv in {args.ckpt / 'teacher.txt'}")
    return [args.count_dir / f"dlsuisho_unique_{n:03d}.{args.kind}.npy" for n in nums]


def select(
    c: np.ndarray, percent: float, by: str, tie_break: str, seed: int
) -> tuple[np.ndarray, dict]:
    possible = np.flatnonzero(~structural_impossible())
    cp = c[possible]
    if tie_break == "random":
        tb = np.random.default_rng(seed).permutation(possible.size)
    else:
        tb = np.arange(possible.size)
    order = np.lexsort((tb, cp))  # 主キー: 出現回数, 副キー: タイ用の順序
    if by == "features":
        k = int(round(percent / 100 * possible.size))
    else:
        cum = np.cumsum(cp[order].astype(np.float64))
        k = int(np.searchsorted(cum, percent / 100 * cum[-1], side="right"))
    chosen = np.sort(possible[order[:k]])
    info: dict = {"n_possible": int(possible.size), "n_selected": k}
    if k:
        b = int(cp[order[k - 1]])
        n_tie = int((cp == b).sum())
        n_tie_in = int((c[chosen] == b).sum())
        info.update(
            max_count_selected=b,
            boundary_ties_total=n_tie,
            boundary_ties_selected=n_tie_in,
            boundary_split=n_tie_in < n_tie,
        )
    return chosen, info


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--arm")
    src.add_argument("--ckpt", type=Path)
    src.add_argument("--counts", type=Path, nargs="+")
    ap.add_argument("--kind", default="both", choices=["both", "dropped_both", "stm", "nstm"])
    ap.add_argument("--count-dir", type=Path, default=COUNT_DIR)
    ap.add_argument("--percent", type=float, required=True)
    ap.add_argument("--by", choices=["features", "occurrences"], default="features")
    ap.add_argument("--tie-break", choices=["random", "index"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--report-counts",
        type=Path,
        nargs="*",
        default=[],
        help="シェアを追加で報告する別のカウント配列 (和)",
    )
    ap.add_argument("--out", type=Path, required=True, help="出力 .npy (int64, 昇順)")
    args = ap.parse_args()

    paths = count_paths(args)
    c = sum_counts(paths)
    chosen, info = select(c, args.percent, args.by, args.tie_break, args.seed)
    tot = float(c.sum())
    info.update(
        count_files=[str(p) for p in paths],
        percent=args.percent,
        by=args.by,
        tie_break=args.tie_break,
        seed=args.seed,
        occurrence_share=float(c[chosen].sum()) / tot if tot else 0.0,
    )
    if args.report_counts:
        r = sum_counts(args.report_counts)
        info["report_counts_files"] = [str(p) for p in args.report_counts]
        info["report_occurrence_share"] = float(r[chosen].sum()) / float(r.sum())
    np.save(args.out, chosen.astype(np.int64))
    Path(str(args.out) + ".json").write_text(json.dumps(info, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in info.items() if k not in ("count_files", "report_counts_files")}
        )
    )


if __name__ == "__main__":
    main()
