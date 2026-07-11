"""nn.bin の重み分布統計と可視化 (docs/PLAN.md フェーズ1)。

使い方:
    .venv/bin/python tools/analyze_nn.py ../suisho11/nn.bin -o experiments/001-nn-analysis

出力: stats.md (統計表) と PNG 図。図は単一系列のみ (凡例不要)、
単一色相 + 控えめなグリッドで描画する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# スクリプト直接実行 (python tools/analyze_nn.py) でも tools パッケージを解決できるようにする
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.nnue import NNUEModel, parse_file  # noqa: E402

HUE = "#2563b0"  # 単一色相 (magnitude 用)
GRID = {"color": "#d4d4d8", "linewidth": 0.6}


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def ft_stats(model: NNUEModel) -> list[str]:
    w = model.feature_transformer.weights
    b = model.feature_transformer.biases
    flat = w.reshape(-1)
    lines = [
        "## FeatureTransformer",
        "",
        f"- weights: shape {w.shape} (feature × channel), int16",
        f"- weights 範囲: [{flat.min()}, {flat.max()}], 平均 {flat.mean():.3f}, "
        f"標準偏差 {flat.std():.2f}",
        f"- biases 範囲: [{b.min()}, {b.max()}], 平均 {b.mean():.1f}",
        "",
        "| 条件 | 割合 |",
        "|---|---|",
    ]
    for thr in (0, 1, 2, 4, 8, 16):
        frac = np.count_nonzero(np.abs(flat) <= thr) / flat.size
        lines.append(f"| \\|w\\| ≤ {thr} | {_pct(frac)} |")
    return lines


def channel_l1(model: NNUEModel) -> np.ndarray:
    w = model.feature_transformer.weights.astype(np.int64)
    return np.abs(w).sum(axis=0)  # (half_dims,) チャネルごとの L1 ノルム


def channel_stats(l1: np.ndarray) -> list[str]:
    order = np.sort(l1)[::-1]
    half = len(order) // 2
    top_half = order[:half].sum()
    return [
        "",
        "## FT 出力チャネル重要度 (L1 ノルム)",
        "",
        f"- チャネル数: {len(l1)}",
        f"- L1 ノルム範囲: [{order[-1]:,}, {order[0]:,}], 中央値 {int(np.median(order)):,}",
        f"- 最大/最小比: {order[0] / max(order[-1], 1):.1f}×",
        f"- 上位 {half} チャネルが全 L1 質量に占める割合: {_pct(top_half / order.sum())}",
    ]


def _logical_in_dims(model: NNUEModel) -> dict[str, int]:
    # パディング列を除いた論理入力次元。ゼロ割合の分母に padding を含めると
    # スパース性を過大評価する (fc_1 は 32 列中 18 列が構造的ゼロ)
    arch = model.arch
    return {"fc_0": arch.half_dims, "fc_1": arch.fc1_in, "fc_2": arch.hidden2_dims}


def stack_stats(model: NNUEModel) -> list[str]:
    lines = [
        "",
        "## LayerStacks (9 スタック)",
        "",
        "| layer | shape (out × in, padding 除外) | 重み範囲 | ゼロ割合 |",
        "|---|---|---|---|",
    ]
    for name, in_dims in _logical_in_dims(model).items():
        ws = np.stack([getattr(s, name).weights for s in model.layer_stacks])[:, :, :in_dims]
        zero = np.count_nonzero(ws == 0) / ws.size
        lines.append(
            f"| {name} | {ws.shape[1]} × {in_dims} | [{ws.min()}, {ws.max()}] | {_pct(zero)} |"
        )
    fc1 = np.stack([s.fc_1.weights for s in model.layer_stacks])
    pad_zero = not fc1[:, :, model.arch.fc1_in :].any()
    lines += [
        "",
        f"- fc_1 のパディング列 (入力 {model.arch.fc1_in}〜{model.arch.fc1_padded_in - 1}) が"
        f"全て 0: {pad_zero}",
    ]
    return lines


def plot_ft_weight_hist(model: NNUEModel, out: Path) -> None:
    flat = np.clip(model.feature_transformer.weights.reshape(-1), -100, 100)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(flat, bins=201, range=(-100.5, 100.5), color=HUE)
    ax.set_yscale("log")
    ax.set_xlabel("weight value (int16, clipped to ±100)")
    ax.set_ylabel("count (log)")
    ax.set_title("FT weights distribution")
    _style_axes(ax)
    _save(fig, out / "ft_weight_hist.png")


def plot_channel_l1(l1: np.ndarray, out: Path) -> None:
    order = np.sort(l1)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(order, color=HUE, linewidth=2)
    ax.set_xlabel("channel rank")
    ax.set_ylabel("L1 norm")
    ax.set_title("FT output channels by L1 norm (descending)")
    _style_axes(ax)
    _save(fig, out / "ft_channel_l1.png")


def plot_fc_weight_hists(model: NNUEModel, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
    for ax, (name, in_dims) in zip(axes, _logical_in_dims(model).items(), strict=True):
        ws = np.concatenate(
            [getattr(s, name).weights[:, :in_dims].reshape(-1) for s in model.layer_stacks]
        )
        ax.hist(ws, bins=64, range=(-128, 128), color=HUE)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("weight (int8)")
        _style_axes(ax)
    axes[0].set_ylabel("count (log)")
    fig.suptitle("LayerStacks weight distributions (9 stacks combined)")
    _save(fig, out / "fc_weight_hists.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nn_bin", type=Path, help="nn.bin のパス")
    parser.add_argument("-o", "--out", type=Path, required=True, help="出力ディレクトリ")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = parse_file(args.nn_bin)
    l1 = channel_l1(model)

    lines = [
        "# nn.bin 統計 (tools/analyze_nn.py 生成)",
        "",
        f"- 入力: `{args.nn_bin}`",
        f"- arch 文字列: `{model.architecture.decode('ascii', 'replace')}`",
        *ft_stats(model),
        *channel_stats(l1),
        *stack_stats(model),
    ]
    (args.out / "stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.save(args.out / "ft_channel_l1.npy", l1)

    plot_ft_weight_hist(model, args.out)
    plot_channel_l1(l1, args.out)
    plot_fc_weight_hists(model, args.out)
    print(f"wrote stats.md and plots to {args.out}")


if __name__ == "__main__":
    main()
