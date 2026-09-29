"""BulletOu checkpoint の FT 行をアブレーションして nn.bin を書き出す (experiment-009)。

checkpoint の state.bin (畳み込み前の f32 重み) から w_specific の指定行を書き換え、
BulletOu 2a8e5ed と同じ手順で畳み込み・量子化・LEB128 符号化して nn.bin を書く。
header と layer stack は checkpoint の nn.bin からバイト単位でコピーする (FT に依存しない)。

モード:
  none             : 書き換えなし (再現性の検証用。checkpoint の nn.bin とバイト単位で一致する)
  zero-specific    : w_specific[f] = 0。w_factor は残す (= 厳密に「未学習」。初期値 ±0.00275 は
                     スケール 127 で 0 に丸まるので、0 と初期値は nn.bin 上で同一)
  random-specific  : w_specific[f] ~ N(0, σ²) (要素ごとに独立)。w_factor は残す。
                     σ = 学習後の w_specific の RMS。基準行 = アブレーション対象外かつ構造的に現れ得る
                     特徴量の行 (--sigma-scope global: 全要素で 1 つの σ / per-column: FT の列ごとの σ)。
                     --sigma で直接指定も可。乱数は numpy PCG64 (--seed)

使い方:
  data/matchenv/bin/python experiments/009-data-scaling/ablate/ablate.py \\
      --ckpt data/bulletou/checkpoints/<run>/0001 --features rare5.npy \\
      --mode zero-specific --out /tmp/x/nn.bin
  --features は .npy (整数配列) か テキスト (空白/改行/カンマ区切りの整数, # 以降はコメント)。
  出力と同じ場所に <out>.json (モード・特徴量数・σ・sha256 等) を書く。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nnbin  # noqa: E402
from nnbin import FT_SIZE, N_BASE  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "feature_count"))
from analyze import structural_impossible  # noqa: E402


def load_features(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        f = np.load(path)
    else:
        text = "\n".join(line.split("#", 1)[0] for line in path.read_text().splitlines())
        f = np.array([int(t) for t in text.replace(",", " ").split()], dtype=np.int64)
    f = np.unique(np.asarray(f, dtype=np.int64).ravel())
    if f.size and (f[0] < 0 or f[-1] >= N_BASE):
        raise ValueError(f"feature index out of range [0, {N_BASE})")
    return f


def random_rows(
    spec: np.ndarray, feats: np.ndarray, scope: str, sigma: float | None, seed: int
) -> tuple[np.ndarray, dict]:
    info: dict = {"dist": "normal(0, sigma^2), iid per element", "seed": seed, "sigma_scope": scope}
    if sigma is None:
        ref = ~structural_impossible()
        ref[feats] = False
        idx = np.flatnonzero(ref)
        if idx.size == 0:
            raise SystemExit(
                "no reference rows left for sigma (all possible features ablated): pass --sigma"
            )
        # RMS を行ブロックごとに f64 で累積 (全体を f64 にしない)
        ss = np.zeros(FT_SIZE, dtype=np.float64)
        s1 = np.zeros(FT_SIZE, dtype=np.float64)
        for i in range(0, idx.size, 8192):
            blk = spec[idx[i : i + 8192]].astype(np.float64)
            ss += (blk * blk).sum(axis=0)
            s1 += blk.sum(axis=0)
        n = idx.size
        info.update(reference_rows=int(n), reference_mean=float(s1.sum() / (n * FT_SIZE)))
        if scope == "global":
            sig = np.full(FT_SIZE, np.sqrt(ss.sum() / (n * FT_SIZE)))
            info["sigma"] = float(sig[0])
        else:
            sig = np.sqrt(ss / n)
            info.update(
                sigma_col_min=float(sig.min()),
                sigma_col_median=float(np.median(sig)),
                sigma_col_max=float(sig.max()),
            )
    else:
        sig = np.full(FT_SIZE, sigma)
        info["sigma"] = sigma
        info["sigma_scope"] = "fixed"
    rng = np.random.default_rng(seed)
    vals = (rng.standard_normal((feats.size, FT_SIZE)) * sig).astype(np.float32)
    return vals, info


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ckpt", type=Path, required=True, help="checkpoint dir (state.bin|weights.bin と nn.bin)"
    )
    ap.add_argument("--mode", choices=["none", "zero-specific", "random-specific"], required=True)
    ap.add_argument(
        "--features", type=Path, help=".npy または テキストの特徴量 index 一覧 (none では不要)"
    )
    ap.add_argument("--out", type=Path, required=True, help="出力 nn.bin のパス")
    ap.add_argument("--seed", type=int, default=0, help="random-specific の乱数シード")
    ap.add_argument("--sigma-scope", choices=["global", "per-column"], default="global")
    ap.add_argument(
        "--sigma", type=float, help="random-specific の σ を直接指定 (既定: 学習後の RMS)"
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="出力をデコードし、対象行以外が元の nn.bin と一致することを確認",
    )
    args = ap.parse_args()

    t0 = time.time()
    state = nnbin.find_state(args.ckpt)
    src = nnbin.parse_nnbin(args.ckpt / "nn.bin")
    l0w, l0b = nnbin.load_ft(state)
    l0w = np.array(l0w)  # 書き換えるので writable なコピー
    feats = np.zeros(0, dtype=np.int64)
    meta: dict = {"ckpt": str(args.ckpt.resolve()), "mode": args.mode, "state_bin": str(state)}
    if args.mode != "none":
        if args.features is None:
            ap.error("--features is required unless --mode none")
        feats = load_features(args.features)
        meta["features_file"] = str(args.features.resolve())
    meta["n_features"] = int(feats.size)

    spec = l0w[:N_BASE]
    if args.mode == "zero-specific":
        spec[feats] = 0.0
    elif args.mode == "random-specific":
        vals, info = random_rows(spec, feats, args.sigma_scope, args.sigma, args.seed)
        spec[feats] = vals
        meta["random"] = info

    folded = nnbin.fold(l0w)
    out_bytes = nnbin.build_nnbin(src, l0b, folded)
    # FT bias は state.bin から再量子化したものが元の nn.bin と一致するはず (同じ時点の重みである確認)
    if out_bytes[len(src.header) : len(src.header) + len(src.l0b_chunk)] != src.l0b_chunk:
        raise SystemExit(
            "l0b re-quantized from state.bin differs from nn.bin: state.bin and nn.bin are not from the same step"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(out_bytes)
    meta["export_seconds"] = round(time.time() - t0, 1)

    if args.mode == "none":
        meta["identical_to_ckpt_nnbin"] = out_bytes == src.raw
        print(f"byte-identical to checkpoint nn.bin: {meta['identical_to_ckpt_nnbin']}")
    if args.verify:
        new = nnbin.parse_nnbin(args.out).l0w()
        old = src.l0w()
        changed = np.flatnonzero((new != old).any(axis=1))
        mask = np.zeros(N_BASE, dtype=bool)
        mask[feats] = True
        v: dict = {
            "rows_changed": int(changed.size),
            "changed_outside_target": int((~mask[changed]).sum()),
        }
        if args.mode == "zero-specific" and feats.size:
            expect = nnbin.quantize_i16(l0w[N_BASE + feats % nnbin.N_VIRTUAL])
            v["target_rows_equal_round_wfactor"] = bool(np.array_equal(new[feats], expect))
        v["decoded_equals_quantized_fold"] = bool(np.array_equal(new, nnbin.quantize_i16(folded)))
        v["header_and_stacks_identical"] = nnbin.parse_nnbin(args.out).tail == src.tail
        meta["verify"] = v
        print("verify:", json.dumps(v))
        if (
            v["changed_outside_target"]
            or not v["header_and_stacks_identical"]
            or not v["decoded_equals_quantized_fold"]
            or v.get("target_rows_equal_round_wfactor") is False
        ):
            raise SystemExit("verification FAILED")

    meta["sha256"] = {
        "out": nnbin.sha256_file(args.out),
        "ckpt_nnbin": nnbin.sha256_file(args.ckpt / "nn.bin"),
    }
    meta["total_seconds"] = round(time.time() - t0, 1)
    Path(str(args.out) + ".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps({k: meta[k] for k in ("mode", "n_features", "export_seconds", "total_seconds")})
    )


if __name__ == "__main__":
    main()
