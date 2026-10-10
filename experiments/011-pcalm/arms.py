"""experiment-011 の arm 定義。arm 名からレシピを組み立てる (段階的な調整で arm を書き足さずに済むように)。

arm 名の文法: ``<規模>-<手法>[-T<n>][-a<α>][-r<ρ>][-eta<η_h> | -etaA<倍率>][-gn | -eps<仮数>e<指数>][-lr<倍率>][-lrmin<倍率>][-rot]``

- 規模: ``s`` = 小規模 (p30 の 9 ファイル, 既定 E=1)、``f`` = 本番規模 (30 ファイル, 既定 E=20)
- 手法: ``bp`` (改造前の BulletOu), ``pcalm`` (``α`` 既定 1), ``pc`` (``α = 0``)
- ``T``: 推論の回数 (pcalm / pc で必須)、``a``: dual の刻み ``α``、``r``: ペナルティ ``ρ`` (既定 1)
- ``eta``: 状態の刻み ``η_h`` を固定する。``etaA``: 自動 (ミニバッチごとに 倍率 / (1.05 × λ_max の推定),
  BulletOu-pcalm の ``--pcalm-eta-auto``)。どちらも省略したら自動 (倍率 1)
- ``eps``: Ranger の epsilon (``eps2.5e9`` = 2.5e-9)。pcalm / pc の既定は ``EPS_PCALM`` の表 (M0 の較正: 1e-7 ×
  FT の勾配の大きさの比 PC-ALM / BP、experiment-011 notes/m0.md)。bp はレシピの既定 (1e-7) のまま
- ``gn``: 勾配の大きさの正規化 (BulletOu-pcalm の ``--pcalm-grad-norm``)。ミニバッチごとに PC-ALM の勾配を、FT の非ゼロ要素の
  mean |g| が ``GRAD_NORM_REF`` (BP の典型値) になるよう定数倍する。Ranger の epsilon はレシピのまま (1e-7)。
  **2026-10-06 以降の pcalm / pc の arm は gn を付ける** (付けない arm は固定 epsilon 表の旧方式で、FT が epsilon に潰れた)
- ``lr``: LR の最大値の倍率 (レシピ 0.000875 に掛ける)、``lrmin``: LR の最小値の倍率 (レシピ 0.00003 に掛ける)
- ``rot``: 教師ファイルの順序を回転 (016 始まり, experiment-009 の full-rot と同じ考え方)

例: ``s-bp-lr1`` (= bp-recipe), ``s-bp-lr1-rot`` (その複製), ``s-pcalm-T4``, ``s-pcalm-T4-a1.5-lr2``,
``s-pc-T16``, ``f-bp-rot`` (= bp-rep, E=16 で回す), ``f-pcalm-T4-a1.5``

使い方:
    python3 experiments/011-pcalm/arms.py s-pcalm-T4-a1.5        # 内容の表示
    python3 experiments/011-pcalm/arms.py --shell s-bp-lr0.7     # run_training.sh 用の変数代入
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOJO = REPO / "data" / "teacher" / "sojo" / "train"
POS_PER_FILE = 488_964_981

LR = 0.000875
LR_MIN = 0.000030
# 小規模の教師 = experiment-009 の p30 (S1 S6 S3)。ファイル番号の昇順
SMALL_FILES = [1, 3, 6, 11, 13, 16, 21, 23, 26]
FULL_FILES = list(range(1, 31))
_TOKEN = re.compile(r"^(T|a|r|etaA|eta|lrmin|lr)([0-9.e-]+)$")
_EPS = re.compile(r"^eps([0-9.]+)e([0-9]+)$")
# BP の FT の勾配の典型的な大きさ: 非ゼロ要素の mean |g| (batch 65,536)。BP の e1 で 4.3e-7、experiment-009 の full-e16 で 3.8e-7
# (初期値では 2.1e-9)。gn の arm はこの値に揃える (experiment-011 notes/m0.md)
GRAD_NORM_REF = 4.0e-7
# Ranger の epsilon の較正 (2026-10-06, M0): BP の e1 (011-s-bp-lr1/0001) の重み、8,192 局面で、FT の勾配の
# 非ゼロ要素の |g| の中央値の比 (PC-ALM / BP) × 1e-7 (BP のレシピの epsilon)。η は自動 (倍率 1)。
# PC-ALM の勾配は BP の 1/100〜1/1000 で、epsilon 1e-7 のままでは FT の更新が epsilon に潰されるため
EPS_PCALM: dict[tuple[int, float, float], float] = {
    (2, 1.0, 1.0): 1.4e-10,
    (3, 1.0, 1.0): 3.7e-10,
    (4, 1.0, 1.0): 6.7e-10,
    (8, 1.0, 1.0): 2.5e-9,
    (16, 1.0, 1.0): 9.6e-9,
    (4, 1.5, 1.0): 8.8e-10,
    (4, 1.0, 2.0): 4.9e-10,
    (8, 1.5, 1.0): 3.5e-9,
    (8, 1.0, 2.0): 1.7e-9,
    (4, 0.0, 1.0): 2.3e-10,
    (8, 0.0, 1.0): 5.3e-10,
    (16, 0.0, 1.0): 1.1e-9,
    (32, 0.0, 1.0): 2.0e-9,
}


@dataclass
class Arm:
    name: str
    scale: str  # "s" | "f"
    method: str  # "bp" | "pcalm" | "pc"
    files: list[int] = field(default_factory=list)
    epochs: int = 1
    lr: float = LR
    lr_min: float = LR_MIN
    steps: int | None = None  # T
    alpha: float | None = None
    rho: float | None = None
    eta_h: float | None = None
    eta_auto: float | None = None
    eps: float | None = None  # Ranger の epsilon (None = レシピの既定)
    grad_norm: float | None = None  # gn: 勾配の大きさの正規化の基準 (None = しない)

    @property
    def credit_args(self) -> list[str]:
        eps = [] if self.eps is None else ["--optimizer-epsilon", repr(self.eps)]
        if self.method == "bp":
            return eps
        return [
            "--credit",
            "pcalm",
            "--pcalm-steps",
            str(self.steps),
            "--pcalm-alpha",
            repr(self.alpha),
            "--pcalm-rho",
            repr(self.rho),
            *(
                ["--pcalm-eta-h", repr(self.eta_h)]
                if self.eta_h is not None
                else ["--pcalm-eta-auto", repr(self.eta_auto)]
            ),
            *([] if self.grad_norm is None else ["--pcalm-grad-norm", repr(self.grad_norm)]),
            *eps,
        ]

    def teacher(self) -> str:
        return ",".join(str(SOJO / f"dlsuisho_unique_{n:03d}.psv") for n in self.files)


def display_name(arm: Arm) -> str:
    """W&B の run 名 (docs/wandb-guide.md §3.1): arm 名 (= checkpoint のフォルダ名から ``011-`` を除いたもの) の規模を書き下したもの。

    例: ``s-bp-lr1`` → ``small-bp-lr1``, ``s-pcalm-T4-a1.5-etaA0.75`` → ``small-pcalm-T4-a1.5-etaA0.75``,
    ``f-bp-rot`` → ``full-bp-rot`` (2026-10-06 ユーザー決定)。全設定は W&B の config に残る。
    """
    scale, rest = arm.name.split("-", 1)
    return f"{'small' if scale == 's' else 'full'}-{rest}"


def rotate(files: list[int], start: int = 16) -> list[int]:
    i = files.index(start)
    return files[i:] + files[:i]


def parse(name: str) -> Arm:
    parts = name.split("-")
    if len(parts) < 2 or parts[0] not in ("s", "f") or parts[1] not in ("bp", "pcalm", "pc"):
        raise ValueError(f"arm 名が不正: {name} (例: s-bp-lr1, s-pcalm-T4)")
    scale, method = parts[0], parts[1]
    arm = Arm(name=name, scale=scale, method=method)
    arm.files = list(SMALL_FILES if scale == "s" else FULL_FILES)
    arm.epochs = 1 if scale == "s" else 20
    rot = False
    for tok in parts[2:]:
        if tok == "rot":
            rot = True
            continue
        if tok == "gn":
            arm.grad_norm = GRAD_NORM_REF
            continue
        me = _EPS.match(tok)
        if me:
            arm.eps = float(me.group(1)) * 10.0 ** -int(me.group(2))
            continue
        m = _TOKEN.match(tok)
        if not m:
            raise ValueError(f"arm 名の要素が不正: {tok} ({name})")
        key, val = m.group(1), m.group(2)
        if key == "T":
            arm.steps = int(val)
        elif key == "a":
            arm.alpha = float(val)
        elif key == "r":
            arm.rho = float(val)
        elif key == "eta":
            arm.eta_h = float(val)
        elif key == "etaA":
            arm.eta_auto = float(val)
        elif key == "lr":
            arm.lr = LR * float(val)
        elif key == "lrmin":
            arm.lr_min = LR_MIN * float(val)
    if rot:
        arm.files = rotate(arm.files)
    if method == "bp":
        if any(
            x is not None
            for x in (arm.steps, arm.alpha, arm.rho, arm.eta_h, arm.eta_auto, arm.grad_norm)
        ):
            raise ValueError(f"bp に T / a / r / eta は付けない: {name}")
    else:
        if arm.steps is None or arm.steps < 1:
            raise ValueError(f"{method} は T<n> (n >= 1) が必要: {name}")
        if method == "pc":
            if arm.alpha not in (None, 0.0):
                raise ValueError(f"pc は α = 0 (a は付けない): {name}")
            arm.alpha = 0.0
        elif arm.alpha is None:
            arm.alpha = 1.0
        if arm.rho is None:
            arm.rho = 1.0
        if arm.eta_h is not None and arm.eta_auto is not None:
            raise ValueError(f"eta と etaA は同時に付けない: {name}")
        if arm.eta_h is None and arm.eta_auto is None:
            arm.eta_auto = 1.0
        if arm.grad_norm is not None and arm.eps is not None:
            raise ValueError(f"gn と eps は同時に付けない (gn はレシピの epsilon を使う): {name}")
        if arm.grad_norm is None and arm.eps is None:
            arm.eps = EPS_PCALM.get((arm.steps, arm.alpha, arm.rho))
        if arm.grad_norm is None and arm.eps is None:
            raise ValueError(
                f"Ranger の epsilon が未較正 (EPS_PCALM に (T={arm.steps}, α={arm.alpha}, ρ={arm.rho}) が無い)。"
                f"eps<仮数>e<指数> を付けるか表に足す: {name}"
            )
    return arm


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("arm")
    ap.add_argument("--shell", action="store_true", help="run_training.sh 用の変数代入を出力")
    args = ap.parse_args()
    arm = parse(args.arm)
    if args.shell:
        print(f"TEACHER={shlex.quote(arm.teacher())}")
        print(f"N_FILES={len(arm.files)}")
        print(f"EPOCHS_DEFAULT={arm.epochs}")
        print(f"LR={arm.lr!r}")
        print(f"LR_MIN={arm.lr_min!r}")
        print(f"METHOD={arm.method}")
        print(f"CREDIT_ARGS=({' '.join(shlex.quote(a) for a in arm.credit_args)})")
    else:
        print(
            json.dumps(
                {**asdict(arm), "credit_args": arm.credit_args, "run_name": display_name(arm)},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
