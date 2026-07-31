"""fc_0 入力の非ゼロマスクからチャネル並べ替え σ を最適化する (experiment-003)。

入力: ft_stats.patch 計測ビルドが書いたマスクファイル
      (1 サンプル = 128 バイト = 1024 bit little-endian、bit j = fc_0 入力 j が非ゼロ)
出力: perm.npy — int32 (512,)。新位置 p に置く旧ペア番号 perm[p]
      (適用は apply_permutation.py が行う)

目的関数: fc_0 sparse カーネルは入力 1024 本を連続 4ch チャンク (int32) 単位で
スキップする。並べ替えの自由度はペア (j, j+512) 単位なので、マスクを前半/後半
(stm/opp 視点) の 512 bit ずつ 2 サンプルに分解し、「チャンク (4 ペア) が全ゼロに
なる割合」を最大化する。貪欲構築 (ゼロ率の高いチャネルを種に共ゼロ最大の仲間を
集める) 後、ランダム 2 チャネル交換の山登りで磨く。

usage: optimize_permutation.py --masks FILE --out perm.npy [--opt-samples 300000]
       [--swap-iters 30000] [--seed 7]
"""

from __future__ import annotations

import argparse

import numpy as np

N_PAIRS = 512
CHUNK = 4
N_CHUNKS = N_PAIRS // CHUNK  # 128


def load_half_masks(path: str) -> np.ndarray:
    """マスクファイル → (2N, 512) bool 行列 (True = 非ゼロ)。前半/後半を別サンプル扱い。"""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 128:
        raise ValueError(f"mask file size {raw.size} is not a multiple of 128")
    bits = np.unpackbits(raw.reshape(-1, 128), axis=1, bitorder="little").astype(bool)
    return np.vstack([bits[:, :N_PAIRS], bits[:, N_PAIRS:]])


def chunk_nonzero_rate(nz: np.ndarray, perm: np.ndarray) -> float:
    """並べ替え perm 適用後のチャンク非ゼロ率 (0..1)。nz: (M,512) bool。"""
    grouped = nz[:, perm].reshape(nz.shape[0], N_CHUNKS, CHUNK)
    return float(grouped.any(axis=2).mean())


def greedy(nz: np.ndarray) -> np.ndarray:
    """ゼロ率最大の未使用チャネルを種に、共ゼロが最大の 3 チャネルを足して 1 チャンクを作る。"""
    zero = ~nz  # (M, 512)
    # bool のまま @ すると論理和に落ちて共ゼロ「数」にならないため float32 に上げる。
    # 列スライスのコピーを避けるため全 512 列との積を一度に取り、使用済み列は -1 で潰す
    zero_f = zero.astype(np.float32)
    zero_count = zero.sum(axis=0).astype(np.int64)
    remaining = np.ones(N_PAIRS, dtype=bool)
    perm = np.empty(N_PAIRS, dtype=np.int32)
    pos = 0
    for _ in range(N_CHUNKS):
        cand = np.where(remaining)[0]
        seed = cand[np.argmax(zero_count[cand])]
        group = [seed]
        remaining[seed] = False
        gz = zero[:, seed].copy()
        for _ in range(CHUNK - 1):
            scores = gz.astype(np.float32) @ zero_f  # 共ゼロサンプル数 (512,)
            scores[~remaining] = -1.0
            best = int(np.argmax(scores))
            group.append(best)
            remaining[best] = False
            gz &= zero[:, best]
        perm[pos : pos + CHUNK] = group
        pos += CHUNK
    return perm


def hill_climb(
    nz: np.ndarray, perm: np.ndarray, iters: int, rng: np.random.Generator
) -> np.ndarray:
    """異なるチャンク間のランダム 2 チャネル交換。改善時のみ採用。"""
    perm = perm.copy()
    # 列アクセス主体なので Fortran-order にして gather を連続読みにする
    zero = np.asfortranarray(~nz)
    # チャンクごとの全ゼロベクトル (M, 128)
    chunk_zero = zero[:, perm].reshape(zero.shape[0], N_CHUNKS, CHUNK).all(axis=2)
    accepted = 0
    for _ in range(iters):
        a, b = rng.integers(0, N_PAIRS, 2)
        ca, cb = a // CHUNK, b // CHUNK
        if ca == cb:
            continue
        pa = perm[ca * CHUNK : (ca + 1) * CHUNK].copy()
        pb = perm[cb * CHUNK : (cb + 1) * CHUNK].copy()
        ia, ib = a % CHUNK, b % CHUNK
        pa[ia], pb[ib] = pb[ib], pa[ia]
        new_a = zero[:, pa].all(axis=1)
        new_b = zero[:, pb].all(axis=1)
        gain = (
            int(new_a.sum())
            + int(new_b.sum())
            - int(chunk_zero[:, ca].sum())
            - int(chunk_zero[:, cb].sum())
        )
        if gain > 0:
            perm[ca * CHUNK : (ca + 1) * CHUNK] = pa
            perm[cb * CHUNK : (cb + 1) * CHUNK] = pb
            chunk_zero[:, ca] = new_a
            chunk_zero[:, cb] = new_b
            accepted += 1
    print(f"hill_climb: accepted {accepted}/{iters} swaps")
    return perm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opt-samples", type=int, default=150_000)
    ap.add_argument("--swap-iters", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    nz_full = load_half_masks(args.masks)
    rng = np.random.default_rng(args.seed)
    print(
        f"samples (half-vectors): {nz_full.shape[0]}, "
        f"channel nonzero rate mean: {nz_full.mean():.4f}"
    )

    if nz_full.shape[0] > args.opt_samples:
        idx = rng.choice(nz_full.shape[0], args.opt_samples, replace=False)
        nz_opt = nz_full[idx]
    else:
        nz_opt = nz_full

    identity = np.arange(N_PAIRS, dtype=np.int32)
    base = chunk_nonzero_rate(nz_full, identity)
    print(f"chunk nonzero rate (identity): {base:.4f}")

    perm = greedy(nz_opt)
    print(f"chunk nonzero rate (greedy, full set): {chunk_nonzero_rate(nz_full, perm):.4f}")

    perm = hill_climb(nz_opt, perm, args.swap_iters, rng)
    final = chunk_nonzero_rate(nz_full, perm)
    print(f"chunk nonzero rate (greedy+swap, full set): {final:.4f}")
    print(f"relative reduction vs identity: {(base - final) / base * 100:.1f}%")

    assert sorted(perm.tolist()) == list(range(N_PAIRS)), "perm is not a permutation"
    np.save(args.out, perm)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
