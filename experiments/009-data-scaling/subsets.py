"""experiment-009 の教師サブセット定義 (奏乗 30 ファイル → 10 サブスタック)。

サブスタック S_d (d = 0..9) = ファイル番号の 1 の位が d の 3 ファイル
    S_1 = {001, 011, 021}, ..., S_0 = {010, 020, 030}
ファイル内・ファイル間のデータ順序 (生成時刻順か等) は文書化されていないため、
連番の塊ではなく 1 の位で交互に取り、各サブスタックが 001〜030 全域にまたがるようにする。

Stage 1 (粗いカーブ) はネスト構造: NESTED_ORDER の先頭 k 個のサブスタックを使う。
    p10 ⊂ p30 ⊂ p50 ⊂ full
p60〜p90 も同じくネスト (NESTED_ORDER の先頭 6〜9 個)。旧案の「連続する k 個を落とす 3 パターン」は廃止。

学習時のファイル順は常にファイル番号の昇順 (full-rot のみ 016 始まりの回転)。

使い方:
    uv run python experiments/009-data-scaling/subsets.py            # 全 arm の一覧
    uv run python experiments/009-data-scaling/subsets.py --teacher p30   # BulletOu --teacher 用のカンマ区切り
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOJO = REPO / "data" / "teacher" / "sojo" / "train"
POS_PER_FILE = 488_964_981  # 029 のみ +7 局面 (無視できる差)
N_FILES = 30

# ダウンロード順もこの順 (小さいサブセットから揃う)
NESTED_ORDER = [1, 6, 3, 8, 4, 2, 9, 5, 7, 0]


def substack(d: int) -> list[int]:
    """1 の位が d のファイル番号 (1..30)。"""
    return [n for n in range(1, N_FILES + 1) if n % 10 == d]


def files_of(stacks: list[int]) -> list[int]:
    return sorted(n for d in stacks for n in substack(d))


ARMS: dict[str, list[int]] = {
    # Stage 1
    "full": files_of(list(range(10))),
    "full-rot": [((n - 1 + 15) % N_FILES) + 1 for n in range(1, N_FILES + 1)],  # 016→030→001→015
    "p50": files_of(NESTED_ORDER[:5]),
    "p30": files_of(NESTED_ORDER[:3]),
    "p10": files_of(NESTED_ORDER[:1]),
}
# p60〜p90 もネスト (2026-09-29 決定): p50 ⊂ p60 ⊂ p70 ⊂ p80 ⊂ p90 ⊂ full。
# S0 (010/020/030) は full 以外の全 arm で held-out になる
for k in (6, 7, 8, 9):
    ARMS[f"p{k * 10}"] = files_of(NESTED_ORDER[:k])
# p3 (2026-10-02 追加): 1 ファイル = 3.3%。p10 の削減 (90%) でも下位 75% のアブレーションほど弱くならない
# 見込みのため、データ削減側の曲線をその先まで延ばす。p3 ⊂ p10
ARMS["p3"] = [1]


def teacher_arg(arm: str) -> str:
    return ",".join(str(SOJO / f"dlsuisho_unique_{n:03d}.psv") for n in ARMS[arm])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", metavar="ARM", help="BulletOu --teacher 文字列を出力")
    ap.add_argument("--files", metavar="ARM", help="ファイル番号をスペース区切りで出力")
    args = ap.parse_args()
    if args.teacher:
        print(teacher_arg(args.teacher))
        return
    if args.files:
        print(" ".join(f"{n:03d}" for n in ARMS[args.files]))
        return
    for arm, files in ARMS.items():
        n = len(files)
        print(
            f"{arm:10s} {n:2d} files  {n * POS_PER_FILE / 1e9:6.2f}B pos  {' '.join(f'{x:03d}' for x in files)}"
        )


if __name__ == "__main__":
    main()
