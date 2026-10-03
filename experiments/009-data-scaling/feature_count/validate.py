"""feature_count (Rust, BulletOu の ShogiHalfKa2 を使用) の特徴インデックスを cshogi で独立に検算する。

Rust 側の `--dump N --dump-stride S` が出す JSON Lines (SFEN + stm/nstm 特徴リスト) を読み、
同じ record を .psv から直接読んで cshogi でデコードし、HalfKA2 の定義式から特徴を再計算して
(1) 盤面 (SFEN) と (2) 特徴の多重集合 (stm / nstm) が一致するかを確認する。

HalfKA2 (YaneuraOu Features::HalfKA2 と同じ):
    index = kb * 1629 + bp
    kb  = 視点側の玉のマス (後手視点は 180 度回転: 80 - sq)
    bp  = BonaPiece。盤上駒は base[駒種][味方/相手] + (回転後の) sq、
          持ち駒 i 枚目は hand_base[駒種][味方/相手] + i - 1、
          玉は自玉・相手玉とも 1548 + (回転後の) sq (相手玉 plane 1629.. を自玉 plane に畳む)

使い方:
    feature_count --dump 5000 --dump-stride 97787 FILE.psv > dump.jsonl
    data/matchenv/bin/python validate.py FILE.psv dump.jsonl
"""

from __future__ import annotations

import json
import sys

import cshogi
import numpy as np

PIECE_INPUTS = 1629
F_KING = 1548
RECORD_SIZE = 40

# 盤上駒の BonaPiece base: 駒種 (cshogi/YaneuraOu の値) -> (相手, 味方)
F_PAWN, E_PAWN = 90, 171
F_LANCE, E_LANCE = 252, 333
F_KNIGHT, E_KNIGHT = 414, 495
F_SILVER, E_SILVER = 576, 657
F_GOLD, E_GOLD = 738, 819
F_BISHOP, E_BISHOP = 900, 981
F_HORSE, E_HORSE = 1062, 1143
F_ROOK, E_ROOK = 1224, 1305
F_DRAGON, E_DRAGON = 1386, 1467
BOARD_BASE = {
    1: (E_PAWN, F_PAWN),
    2: (E_LANCE, F_LANCE),
    3: (E_KNIGHT, F_KNIGHT),
    4: (E_SILVER, F_SILVER),
    5: (E_BISHOP, F_BISHOP),
    6: (E_ROOK, F_ROOK),
    7: (E_GOLD, F_GOLD),
    9: (E_GOLD, F_GOLD),  # と
    10: (E_GOLD, F_GOLD),  # 成香
    11: (E_GOLD, F_GOLD),  # 成桂
    12: (E_GOLD, F_GOLD),  # 成銀
    13: (E_HORSE, F_HORSE),
    14: (E_DRAGON, F_DRAGON),
}
# 持ち駒 base: cshogi の pieces_in_hand の並び (歩 香 桂 銀 金 角 飛) -> (相手, 味方)
HAND_BASE = [(20, 1), (44, 39), (54, 49), (64, 59), (74, 69), (82, 79), (88, 85)]


def halfka2_features(board: cshogi.Board, persp: int) -> list[int]:
    """cshogi の局面から視点 persp (0=先手, 1=後手) の HalfKA2 特徴を計算する。"""

    def rot(sq: int) -> int:
        return sq if persp == 0 else 80 - sq

    kb = rot(board.king_square(persp))
    feats = []
    for sq, pc in enumerate(board.pieces):
        if pc == 0:
            continue
        color, pt = pc >> 4, pc & 15
        if pt == 8:  # 玉: 自玉・相手玉とも F_KING plane
            feats.append(kb * PIECE_INPUTS + F_KING + rot(sq))
            continue
        bp = BOARD_BASE[pt][int(color == persp)] + rot(sq)
        feats.append(kb * PIECE_INPUTS + bp)
    for owner in (0, 1):
        for k, n in enumerate(board.pieces_in_hand[owner]):
            base = HAND_BASE[k][int(owner == persp)]
            for i in range(1, n + 1):
                feats.append(kb * PIECE_INPUTS + base + i - 1)
    return feats


def main() -> None:
    psv_path, dump_path = sys.argv[1], sys.argv[2]
    mm = np.memmap(psv_path, dtype=np.uint8, mode="r")
    board = cshogi.Board()
    n = bad_sfen = bad_feat = 0
    for line in open(dump_path):
        d = json.loads(line)
        rec = np.array(mm[d["index"] * RECORD_SIZE : d["index"] * RECORD_SIZE + 32])
        board.set_psfen(rec)
        # 盤面一致: Rust の SFEN を cshogi に読ませて正規化し比較 (手数は無視)
        ref = board.sfen().rsplit(" ", 1)[0]
        got = cshogi.Board(d["sfen"]).sfen().rsplit(" ", 1)[0]
        if ref != got:
            bad_sfen += 1
            print(f"SFEN mismatch @{d['index']}: rust={got} cshogi={ref}")
        stm = board.turn
        exp_stm = sorted(halfka2_features(board, stm))
        exp_nstm = sorted(halfka2_features(board, 1 - stm))
        if exp_stm != sorted(d["stm"]) or exp_nstm != sorted(d["nstm"]):
            bad_feat += 1
            if bad_feat <= 5:
                print(f"feature mismatch @{d['index']} {ref}")
                print(
                    "  stm  only-rust",
                    sorted(set(d["stm"]) - set(exp_stm)),
                    "only-py",
                    sorted(set(exp_stm) - set(d["stm"])),
                )
                print(
                    "  nstm only-rust",
                    sorted(set(d["nstm"]) - set(exp_nstm)),
                    "only-py",
                    sorted(set(exp_nstm) - set(d["nstm"])),
                )
        n += 1
    print(f"checked {n} records: sfen mismatches={bad_sfen}, feature mismatches={bad_feat}")
    sys.exit(1 if bad_sfen or bad_feat else 0)


if __name__ == "__main__":
    main()
