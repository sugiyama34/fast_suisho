"""feature_count の出力 (.npy) の要約: ゼロ件数・対数ビンのヒストグラム・上位/下位特徴のデコード。

HalfKA2 index = kb * 1629 + bp を人が読める形に戻す。座標は視点側から見た (後手視点は 180 度回転済みの)
マスで、「味方/相手」は視点側から見た駒の持ち主。

構造的に出現し得ない特徴 (盤上駒が自玉と同じマス、行き所のない歩・香・桂、
持ち駒領域の空きスロット (bp=0 と各駒種の予備)、相手玉が自玉に隣接) を別枠で数える。

使い方:
    data/matchenv/bin/python analyze.py /mnt/nvme1/sugiyama/feature_counts/dlsuisho_unique_001.both.npy [...more.npy]
    (複数指定すると合算 = arm のカウント)
"""

from __future__ import annotations

import sys

import numpy as np

PIECE_INPUTS = 1629
N_KB = 81
F_KING = 1548
KANJI_NUM = "一二三四五六七八九"
# (base, 名前, 味方か)。盤上駒は 81 マス、持ち駒は枚数
BOARD_PLANES = [
    (90, "歩", True), (171, "歩", False), (252, "香", True), (333, "香", False),
    (414, "桂", True), (495, "桂", False), (576, "銀", True), (657, "銀", False),
    (738, "金", True), (819, "金", False), (900, "角", True), (981, "角", False),
    (1062, "馬", True), (1143, "馬", False), (1224, "飛", True), (1305, "飛", False),
    (1386, "龍", True), (1467, "龍", False),
]  # fmt: skip
HAND_PLANES = [
    (1, 18, "歩", True), (20, 18, "歩", False), (39, 4, "香", True), (44, 4, "香", False),
    (49, 4, "桂", True), (54, 4, "桂", False), (59, 4, "銀", True), (64, 4, "銀", False),
    (69, 4, "金", True), (74, 4, "金", False), (79, 2, "角", True), (82, 2, "角", False),
    (85, 2, "飛", True), (88, 2, "飛", False),
]  # fmt: skip


def sq_name(sq: int) -> str:
    return f"{sq // 9 + 1}{KANJI_NUM[sq % 9]}"


def decode(idx: int) -> str:
    kb, bp = divmod(idx, PIECE_INPUTS)
    head = f"自玉{sq_name(kb)}"
    if bp >= F_KING:
        s = bp - F_KING
        return f"{head} | {'自玉' if s == kb else '相手玉'}{sq_name(s)}"
    if bp >= 90:
        for base, name, friend in reversed(BOARD_PLANES):
            if bp >= base:
                return f"{head} | {'味方' if friend else '相手'}{name}{sq_name(bp - base)}"
    for base, n, name, friend in HAND_PLANES:
        if base <= bp < base + n:
            return f"{head} | {'味方' if friend else '相手'}持ち駒{name}{bp - base + 1}枚目"
    return f"{head} | bp={bp} (持ち駒領域の空きスロット)"


def structural_impossible() -> np.ndarray:
    """合法局面では出現し得ない特徴の mask。"""
    m = np.zeros(N_KB * PIECE_INPUTS, dtype=bool)
    for kb in range(N_KB):
        o = kb * PIECE_INPUTS
        # 持ち駒領域: 使われる枚数スロット以外 (bp=0 と各駒種の予備スロット) は空き
        m[o : o + 90] = True
        for base, n, _, _ in HAND_PLANES:
            m[o + base : o + base + n] = False
        for base, name, friend in BOARD_PLANES:
            m[o + base + kb] = True  # 自玉と同じマス
            for sq in range(81):
                r = sq % 9 if friend else 8 - sq % 9  # その駒から見た段 (0 = 敵陣最奥)
                if name in ("歩", "香") and r == 0 or name == "桂" and r <= 1:
                    m[o + base + sq] = True  # 行き所のない駒
        for s in range(81):
            # 相手玉が自玉に隣接 (s == kb は自玉特徴そのもの)
            if s != kb and abs(s // 9 - kb // 9) <= 1 and abs(s % 9 - kb % 9) <= 1:
                m[o + F_KING + s] = True
    return m


def main() -> None:
    c = sum(np.load(p).astype(np.uint64) for p in sys.argv[1:])
    imp = structural_impossible()
    zero = c == 0
    print(f"features={c.size}  total_occurrences={int(c.sum()):,}")
    print(f"count==0: {int(zero.sum())}  (structurally impossible: {int(imp.sum())}, "
          f"of which zero: {int((zero & imp).sum())}; possible-but-unseen: {int((zero & ~imp).sum())}; "
          f"impossible-but-seen: {int((~zero & imp).sum())})")  # fmt: skip

    pos = c[(c > 0) & ~imp]
    print("\nlog10 histogram of nonzero counts (structurally possible features):")
    edges = [1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = int(((pos >= lo) & (pos < hi)).sum())
        if n:
            print(
                f"  [{lo:>8.0e}, {hi:>8.0e}): {n:>7}  {'#' * max(1, round(60 * n / pos.size)) if n else ''}"
            )
    q = np.percentile(pos, [1, 10, 50, 90, 99])
    print("  percentiles p1/p10/p50/p90/p99:", " / ".join(f"{x:,.0f}" for x in q))

    order = np.argsort(c, kind="stable")
    print("\ntop 15:")
    for i in order[::-1][:15]:
        print(f"  {int(c[i]):>15,}  #{i:<6} {decode(int(i))}")
    nz = [i for i in order if c[i] > 0]
    print("\nbottom 15 nonzero:")
    for i in nz[:15]:
        print(f"  {int(c[i]):>15,}  #{i:<6} {decode(int(i))}")
    unseen = [i for i in order if c[i] == 0 and not imp[i]]
    rng = np.random.default_rng(0)
    print("\nsample of 15 possible-but-unseen:")
    for i in sorted(rng.choice(unseen, size=min(15, len(unseen)), replace=False)):
        print(f"  #{i:<6} {decode(int(i))}")

    # 自玉マス別の総出現 (king bucket の偏り)
    per_kb = c.reshape(N_KB, PIECE_INPUTS).sum(axis=1)
    kb_order = np.argsort(per_kb)[::-1]
    share = per_kb / per_kb.sum()
    print("\nking-bucket share (top 5 / bottom 5):")
    print("  " + ", ".join(f"{sq_name(k)} {share[k]:.2%}" for k in kb_order[:5]))
    print("  " + ", ".join(f"{sq_name(k)} {share[k]:.2e}" for k in kb_order[-5:]))


if __name__ == "__main__":
    main()
