"""チャネル並べ替え σ を nn.bin に適用する (experiment-003)。

等価変換 (report.md「等価変換の定義」参照):
- idx = concat(perm, perm + 512)
- FT weights (131949, 1024) の列と biases (1024,) を idx で置換
- 全 9 スタックの fc_0 weights (8, 1024) の入力列を idx で置換
- 他は一切変更しない (hash / arch 文字列 / fc_1 / fc_2 は元のバイト列のまま)

書き出し前に、ランダム accumulator に対して pairwise-mul → fc_0 をモデル通りに
シミュレートし、置換前後で出力が一致することを全スタックで確認する
(FT 側の列置換と fc_0 側の列置換のペア対応 (j, j+512) が食い違っていれば不一致になる)。
最終的な正しさは検証用固定局面での評価値完全一致で担保する。

usage: apply_permutation.py --nn ../suisho11/nn.bin --perm perm.npy --out /path/to/out/nn.bin
"""

from __future__ import annotations

import argparse

import numpy as np

from tools.nnue import io


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn", required=True)
    ap.add_argument("--perm", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    perm = np.load(args.perm)
    assert perm.shape == (512,) and sorted(perm.tolist()) == list(range(512))
    idx = np.concatenate([perm, perm + 512])

    model = io.parse_file(args.nn)
    ft = model.feature_transformer

    # 等価性チェック: ランダム accumulator (視点ごと 1024ch) から pairwise-mul →
    # fc_0 をモデル通りに計算し、「元の重み」と「FT 列置換後の accumulator +
    # fc_0 列置換後の重み」で出力が一致することを確認する。
    # 注意: 単なる W[:, idx] @ z[idx] == W @ z は任意の置換で恒真なので検証にならない。
    # pairwise-mul を挟むことで、FT 側 (ペア j, j+512 を同時に動かす) と fc_0 側の
    # 置換のペア対応が食い違っていれば不一致として検出される
    def pairwise(acc: np.ndarray) -> np.ndarray:
        u = np.clip(acc[:512], 0, 127)
        v = np.clip(acc[512:], 0, 127)
        return (u * v) // 128

    rng = np.random.default_rng(0)
    acc_stm = rng.integers(-300, 300, size=1024).astype(np.int64)
    acc_opp = rng.integers(-300, 300, size=1024).astype(np.int64)
    z_ref = np.concatenate([pairwise(acc_stm), pairwise(acc_opp)])
    # FT 列置換後の accumulator は acc[idx] (両視点とも同じ σ)
    z_new = np.concatenate([pairwise(acc_stm[idx]), pairwise(acc_opp[idx])])
    for k, stack in enumerate(model.layer_stacks):
        w = stack.fc_0.weights.astype(np.int64)
        ref = w @ z_ref
        new = w[:, idx] @ z_new
        assert np.array_equal(ref, new), f"fc_0 equivalence check failed at stack {k}"

    ft.biases = ft.biases[idx]
    ft.weights = ft.weights[:, idx]
    for stack in model.layer_stacks:
        stack.fc_0.weights = stack.fc_0.weights[:, idx]

    # numpy の fancy-indexing は非連続 view を作らないが、serializer は dtype/shape
    # しか見ないため ascontiguousarray で確実に実体化しておく
    ft.biases = np.ascontiguousarray(ft.biases)
    ft.weights = np.ascontiguousarray(ft.weights)
    for stack in model.layer_stacks:
        stack.fc_0.weights = np.ascontiguousarray(stack.fc_0.weights)

    io.serialize_file(model, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
