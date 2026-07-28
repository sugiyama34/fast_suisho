"""match_sprt / cshogi API の事前セルフテスト (エンジン不要)。

uv run python experiments/005-bulletou-sojo/match_selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cshogi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from match_sprt import PentanomialSprt, elo_to_score, score_to_elo  # noqa: E402

HERE = Path(__file__).resolve().parent


def test_sprt_math() -> None:
    assert abs(elo_to_score(0.0) - 0.5) < 1e-12
    assert abs(score_to_elo(elo_to_score(-30.0)) + 30.0) < 1e-9

    # 引き分けペアのみ (分散 0) でも、平均が H1 側なら分散下限により H1 受理へ動く
    s = PentanomialSprt(elo0=-30, elo1=0)
    for _ in range(200):
        s.add_pair(1.0)
    assert s.llr() > 0
    assert s.status() == "accept_h1", s.summary()
    s = PentanomialSprt(elo0=-30, elo1=0)
    for _ in range(100):
        s.add_pair(1.5)
        s.add_pair(0.5)
        s.add_pair(1.0)
    m, v = s.mean_var()
    assert abs(m - 0.5) < 1e-12 and v > 0
    assert s.llr() > 0  # 実測五分 > H0(-30) と H1(0) の中点 (-15 相当) → H1 寄り
    # 一方的に負けるペアを積むと H0 受理
    s = PentanomialSprt(elo0=-30, elo1=0)
    for _ in range(300):
        s.add_pair(0.5)
    assert s.status() == "accept_h0", s.summary()
    print("sprt math: ok", s.summary())


def test_cshogi_api() -> None:
    book = HERE / "start_sfens_500.txt"
    sfen = book.read_text().splitlines()[0].removeprefix("sfen ").strip()
    board = cshogi.Board(sfen=sfen)
    assert board.turn == cshogi.WHITE, "24手目局面は後手番のはず"
    legal = list(board.legal_moves)
    assert legal, "合法手がある"
    mv_usi = cshogi.move_to_usi(legal[0])
    mv = board.move_from_usi(mv_usi)
    assert mv in legal
    board.push(mv)
    assert board.turn == cshogi.BLACK
    assert board.is_draw() == cshogi.NOT_REPETITION
    assert not board.is_game_over()
    # 千日手系の定数が存在することの確認 (match_runner が参照する)
    for name in (
        "REPETITION_DRAW",
        "REPETITION_WIN",
        "REPETITION_LOSE",
        "REPETITION_SUPERIOR",
        "REPETITION_INFERIOR",
        "NOT_REPETITION",
    ):
        assert hasattr(cshogi, name), name
    print("cshogi api: ok (sfen 1 局面で指し手適用・判定関数を確認)")


if __name__ == "__main__":
    test_sprt_math()
    test_cshogi_api()
    print("selftest: all ok")
