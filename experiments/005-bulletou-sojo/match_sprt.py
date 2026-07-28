"""ペア単位 (pentanomial) の SPRT / Elo 信頼区間の計算。

docs/PLAN.md フェーズ3 の方針に従い、1000 局 = 500 ペア (同一開始局面の先後入替)
を独立ペアの列として扱う。trinomial + 独立仮定は分散を過小評価するため使わない。

- ペアスコア: 候補側から見た 2 局の合計得点 s ∈ {0, 0.5, 1, 1.5, 2} を
  x = s/2 ∈ [0,1] に正規化して扱う
- SPRT は GSPRT 近似 (Michel Van den Bergh, fishtest 方式):
    LLR ≈ N (s1 - s0) (2 m - s0 - s1) / (2 v)
  ここで m, v は x の標本平均・標本分散、s0/s1 は H0/H1 の Elo に対応する期待得点
- Elo ↔ 得点: E = 1 / (1 + 10^(-elo/400)) (logistic)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ペアスコア (2 局合計, 候補側) → pentanomial ビン index
PAIR_BINS = (0.0, 0.5, 1.0, 1.5, 2.0)


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score: float) -> float:
    eps = 1e-9
    score = min(max(score, eps), 1.0 - eps)
    return -400.0 * math.log10(1.0 / score - 1.0)


@dataclass
class PentanomialSprt:
    """ペア結果を蓄積し、LLR / Elo / 95% CI を返す。"""

    elo0: float = -30.0
    elo1: float = 0.0
    alpha: float = 0.05
    beta: float = 0.05
    # 分散下限 (llr 参照) と少数ペアの組み合わせで即断しないための最小ペア数。
    # ペア分散が安定するまで判定を保留する
    min_pairs: int = 32
    counts: list[int] = field(default_factory=lambda: [0] * 5)

    def add_pair(self, pair_score: float) -> None:
        self.counts[PAIR_BINS.index(pair_score)] += 1

    @property
    def n_pairs(self) -> int:
        return sum(self.counts)

    def mean_var(self) -> tuple[float, float]:
        """正規化ペアスコア x = s/2 の標本平均と標本分散。"""
        n = self.n_pairs
        if n == 0:
            return 0.5, 0.0
        xs = [b / 2.0 for b in PAIR_BINS]
        m = sum(c * x for c, x in zip(self.counts, xs)) / n
        v = sum(c * (x - m) ** 2 for c, x in zip(self.counts, xs)) / n
        return m, v

    def llr(self) -> float:
        n = self.n_pairs
        m, v = self.mean_var()
        if n == 0:
            return 0.0
        # 全ペア同一結果だと標本分散が 0 になり LLR が計算不能。平均が仮説から
        # 明確に離れているのに永久に continue になるのを防ぐため下限を入れる
        v = max(v, 1e-6)
        s0 = elo_to_score(self.elo0)
        s1 = elo_to_score(self.elo1)
        return n * (s1 - s0) * (2.0 * m - s0 - s1) / (2.0 * v)

    def bounds(self) -> tuple[float, float]:
        lower = math.log(self.beta / (1.0 - self.alpha))
        upper = math.log((1.0 - self.beta) / self.alpha)
        return lower, upper

    def status(self) -> str:
        """'accept_h1' / 'accept_h0' / 'continue'"""
        if self.n_pairs < self.min_pairs:
            return "continue"
        lo, hi = self.bounds()
        llr = self.llr()
        if llr >= hi:
            return "accept_h1"
        if llr <= lo:
            return "accept_h0"
        return "continue"

    def elo_ci(self) -> tuple[float, float, float]:
        """(elo 点推定, 95% CI 下限, 上限)。ペア分散ベース。

        全ペア同一結果だと標本分散 0 → 幅 0 の「CI」になり確信を過大表示する
        ため、llr() と同じ分散下限を適用する (それでも有限標本の CI としては
        楽観的であることに変わりはない — 報告時は n も併記すること)。
        """
        n = self.n_pairs
        m, v = self.mean_var()
        if n == 0:
            return 0.0, -math.inf, math.inf
        se = math.sqrt(max(v, 1e-6) / n)
        return (
            score_to_elo(m),
            score_to_elo(m - 1.96 * se),
            score_to_elo(m + 1.96 * se),
        )

    def summary(self) -> dict:
        elo, lo, hi = self.elo_ci()
        return {
            "pairs": self.n_pairs,
            "pentanomial": list(self.counts),
            "score": round(self.mean_var()[0], 5),
            "elo": round(elo, 2),
            "elo_ci95": [round(lo, 2), round(hi, 2)],
            "llr": round(self.llr(), 3),
            "llr_bounds": [round(b, 3) for b in self.bounds()],
            "sprt": self.status(),
            "h": {"elo0": self.elo0, "elo1": self.elo1, "alpha": self.alpha, "beta": self.beta},
        }
