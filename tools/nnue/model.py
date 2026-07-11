"""Suisho 11 (SFNN_halfka2_1024-7-64-k3k3) nn.bin のデータモデル。

やねうら王 V9.60 の読み込みコード (`evaluate_nnue.cpp` / `nnue_feature_transformer.h` /
`layers/affine_transform*_explicit.h`) が定義するオンディスク形式に対応する。
ファイルに格納されている hash 値は検証せず verbatim に保持する
(Suisho 11 のヘッダ hash はエンジン期待値と不一致だが無害 —
experiments/000-baseline/report.md 既知の注意点 1 を参照)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Arch:
    """ネットワーク次元。ファイルには次元情報が入っていないため外から与える。"""

    input_dims: int = 131949  # HalfKA2<Friend>: SQ_NB(81) * fe_end(1629)
    half_dims: int = 1024  # FeatureTransformer 出力 (片側)
    hidden1_dims: int = 7  # fc_0 出力は hidden1 + 1 (shortcut 項)
    hidden2_dims: int = 64
    layer_stacks: int = 9

    # kMaxSimdWidth = 32 での CeilToMultiple (入力次元のパディング)
    @property
    def fc0_out(self) -> int:
        return self.hidden1_dims + 1

    @property
    def fc0_padded_in(self) -> int:
        return _ceil_to_multiple(self.half_dims, 32)

    @property
    def fc1_in(self) -> int:
        return self.hidden1_dims * 2

    @property
    def fc1_padded_in(self) -> int:
        return _ceil_to_multiple(self.fc1_in, 32)

    @property
    def fc2_padded_in(self) -> int:
        return _ceil_to_multiple(self.hidden2_dims, 32)


def _ceil_to_multiple(n: int, base: int) -> int:
    return (n + base - 1) // base * base


SUISHO11_ARCH = Arch()

# nnue_common.h kVersion
NNUE_VERSION = 0x7AF32F16


@dataclass
class FeatureTransformer:
    """FT 部。ファイル上は LEB128 圧縮された int16 列 (biases → weights)。

    weights の disk 順は feature-major: shape (input_dims, half_dims)。
    """

    hash: int
    biases: np.ndarray  # int16 (half_dims,)
    weights: np.ndarray  # int16 (input_dims, half_dims)


@dataclass
class AffineLayer:
    """全結合層。ファイル上は int32 biases → int8 weights (out, padded_in) row-major。

    padded_in > 論理入力次元 の列はパディング (Suisho 11 では全て 0)。
    """

    biases: np.ndarray  # int32 (out,)
    weights: np.ndarray  # int8 (out, padded_in)


@dataclass
class LayerStack:
    """1 スタック分 (fc_0 → ac_0/ac_sqr_0 → fc_1 → ac_1 → fc_2)。活性化層は無パラメータ。"""

    hash: int
    fc_0: AffineLayer  # (hidden1+1, half_dims)
    fc_1: AffineLayer  # (hidden2, pad32(hidden1*2))
    fc_2: AffineLayer  # (1, pad32(hidden2))


@dataclass
class NNUEModel:
    version: int
    hash: int
    architecture: bytes  # arch 文字列 (非 NUL 終端、格納バイト列そのまま)
    feature_transformer: FeatureTransformer
    layer_stacks: list[LayerStack]
    arch: Arch = field(default_factory=Arch)
