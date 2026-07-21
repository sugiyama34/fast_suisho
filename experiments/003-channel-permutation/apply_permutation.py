"""チャネル並べ替え σ を nn.bin に適用する (experiment-003)。

等価変換 (report.md「等価変換の定義」参照):
- idx = concat(perm, perm + 512)
- FT weights (131949, 1024) の列と biases (1024,) を idx で置換
- 全 9 スタックの fc_0 weights (8, 1024) の入力列を idx で置換
- 他は一切変更しない (hash / arch 文字列 / fc_1 / fc_2 は元のバイト列のまま)

書き出し前に、ランダム入力に対する fc_0 出力の代数一致 (W_new @ z[idx] == W @ z) を
全スタックで確認する。最終的な正しさは検証用固定局面での評価値完全一致で担保する。

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

    # 代数チェック: 置換前の fc_0 で計算した出力と、置換後の fc_0 に置換後入力を
    # 与えた出力が一致すること
    rng = np.random.default_rng(0)
    z = rng.integers(0, 127, size=1024, dtype=np.int64)
    z[rng.random(1024) < 0.6] = 0  # 実際の入力同様スパースに
    for k, stack in enumerate(model.layer_stacks):
        w = stack.fc_0.weights.astype(np.int64)
        ref = w @ z
        new = w[:, idx] @ z[idx]
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
