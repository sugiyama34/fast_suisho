"""loss 測定用の固定標本 (全 arm・全ネット共通) を作る。

一様標本 (各 1,000,000 record, ファイル集合の連結から非復元一様抽出, 固定シード):
  A_s1: S1 = 001/011/021 (全 arm が学習に使う = train loss)
  B_s0: S0 = 010/020/030 (full 以外の全 arm で held-out)
稀な層を厚くした層別標本 (ファイル集合を全走査し、min_count の log10 ビンごとに最大 100,000 record を一様抽出。
|score| >= 32000 は除く。ビン内では一様なので、record ごとの重み design_weight = 母集団のビンの大きさ / 標本数
で重み付けすれば母集団の値の不偏推定になる):
  C_s1_rare: S1 から
  D_s0_rare: S0 から

出力 (--out-dir, 既定 /mnt/nvme1/sugiyama/loss_samples/):
  <name>.psv        PackedSfenValue 40 B × N (元ファイルでの位置の昇順)
  <name>.index.npy  u64 [N]  ファイル集合の連結での record 番号 (file k の先頭 = k 番目までの record 数の和)
  <name>.feat.npy   u32 [N, 80] HalfKA2 特徴 (手番側 40 + 非手番側 40)。loss_eval features の出力
  <name>.bucket.npy u8 [N]   layer stack 番号 (k3k3)
  <name>.meta.npz   record ごとの層別用の値:
                      score (i16), result (i8), ply (u16), used (bool: |score| < 32000 = loss 重み > 0),
                      min_count (u64: 80 個の特徴の full 教師での出現回数の最小値),
                      argmin_feature (u32: その特徴 index),
                      n_lt_1e2 / n_lt_1e3 / n_lt_1e4 / n_lt_1e5 (u8: 出現回数が閾値未満の特徴の個数, 両視点 80 個中),
                      design_weight (f64: 一様標本は 1)
  <name>.json       抽出条件・件数・sha256 (層別標本は <name>.scan.json に母集団のビンの大きさ)
出現回数 = full_counts.both.npy = /mnt/nvme1/sugiyama/feature_counts/dlsuisho_unique_{001..030}.both.npy の和
(loss 重み > 0 の record だけの両視点合計, feature-counts.md)。

使い方:
    data/matchenv/bin/python experiments/009-data-scaling/loss_eval/make_samples.py [--only A_s1]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from subsets import SOJO  # noqa: E402

TOOL = HERE / "target" / "release" / "loss_eval"
COUNT_DIR = Path("/mnt/nvme1/sugiyama/feature_counts")
OUT_DIR = Path("/mnt/nvme1/sugiyama/loss_samples")
RECORD = 40
SCORE_DROP_ABS = 32000

SETS = {
    "A_s1": {
        "files": [1, 11, 21],
        "seed": 20260930,
        "role": "S1: 全 arm が学習に使う (train loss), 一様",
    },
    "B_s0": {
        "files": [10, 20, 30],
        "seed": 20260931,
        "role": "S0: full 以外の全 arm で held-out, 一様",
    },
    "C_s1_rare": {
        "files": [1, 11, 21],
        "seed": 20260932,
        "per_bin": 100_000,
        "role": "S1, min_count の log10 ビンごとの層別 (稀な層を厚く)",
    },
    "D_s0_rare": {
        "files": [10, 20, 30],
        "seed": 20260933,
        "per_bin": 100_000,
        "role": "S0, min_count の log10 ビンごとの層別 (稀な層を厚く)",
    },
}
THRESHOLDS = {"n_lt_1e2": 1e2, "n_lt_1e3": 1e3, "n_lt_1e4": 1e4, "n_lt_1e5": 1e5}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def full_counts(out_dir: Path) -> tuple[np.ndarray, Path]:
    paths = [COUNT_DIR / f"dlsuisho_unique_{n:03d}.both.npy" for n in range(1, 31)]
    c = sum(np.load(p).astype(np.uint64) for p in paths)
    path = out_dir / "full_counts.both.npy"
    np.save(path, c)
    return c, path


def rarity(feat: np.ndarray, counts: np.ndarray, chunk: int = 100_000) -> dict[str, np.ndarray]:
    n = feat.shape[0]
    out = {
        "min_count": np.empty(n, np.uint64),
        "argmin_feature": np.empty(n, np.uint32),
        **{k: np.empty(n, np.uint8) for k in THRESHOLDS},
    }
    big = np.uint64(np.iinfo(np.uint64).max)
    for s in range(0, n, chunk):
        f = feat[s : s + chunk]
        valid = f != np.iinfo(np.uint32).max
        c = np.where(valid, counts[np.where(valid, f, 0)], big)
        am = c.argmin(axis=1)
        out["min_count"][s : s + chunk] = c[np.arange(len(f)), am]
        out["argmin_feature"][s : s + chunk] = f[np.arange(len(f)), am]
        for k, t in THRESHOLDS.items():
            out[k][s : s + chunk] = (c < t).sum(axis=1)
    return out


def log10_bin(mc: np.ndarray) -> np.ndarray:
    """floor(log10(min_count)) を整数比較で (Rust の log10_bin と同じ, 0..9, 10^9 以上は 9)。"""
    thr = np.array([10**k for k in range(1, 10)], dtype=np.uint64)
    return np.searchsorted(thr, mc.astype(np.uint64), side="right").astype(np.int64)


def make(
    name: str,
    spec: dict,
    n: int,
    counts: np.ndarray,
    counts_path: Path,
    out_dir: Path,
    threads: int,
) -> None:
    paths = [SOJO / f"dlsuisho_unique_{k:03d}.psv" for k in spec["files"]]
    sizes = [p.stat().st_size for p in paths]
    assert all(s % RECORD == 0 for s in sizes)
    recs = [s // RECORD for s in sizes]
    starts = np.concatenate([[0], np.cumsum(recs)]).astype(np.int64)
    total = int(starts[-1])

    scan = None
    if "per_bin" in spec:
        prefix = out_dir / name
        cmd = [
            str(TOOL),
            "scan",
            "--files",
            ",".join(map(str, paths)),
            "--counts",
            str(counts_path),
        ]
        cmd += [
            "--per-bin",
            str(spec["per_bin"]),
            "--seed",
            str(spec["seed"]),
            "--out-prefix",
            str(prefix),
        ]
        cmd += ["--threads", str(threads)]
        subprocess.run(cmd, check=True)
        idx = np.load(out_dir / f"{name}.index.npy")
        scan = json.loads((out_dir / f"{name}.scan.json").read_text())
        sampling = (
            f"loss_eval scan: 全 record を走査し |score| < {SCORE_DROP_ABS} の record を min_count の log10 ビンに分け、"
            f"ビンごとに splitmix64(seed ^ splitmix64(index)) が小さい {spec['per_bin']} 個 (seed={spec['seed']})"
        )
    else:
        rng = np.random.default_rng(spec["seed"])
        idx = np.sort(rng.choice(total, size=n, replace=False)).astype(np.uint64)
        np.save(out_dir / f"{name}.index.npy", idx)
        sampling = (
            f"numpy {np.__version__} Generator(PCG64, seed={spec['seed']}).choice(population, n, replace=False),"
            " 昇順に並べ替え"
        )

    psv_path = out_dir / f"{name}.psv"
    with open(psv_path, "wb") as w:
        for k, p in enumerate(paths):
            sel = idx[(idx >= starts[k]) & (idx < starts[k + 1])].astype(np.int64) - starts[k]
            mm = np.memmap(p, dtype=np.uint8, mode="r").reshape(-1, RECORD)
            mm[sel].tofile(w)
            del mm

    feat_path = out_dir / f"{name}.feat.npy"
    bucket_path = out_dir / f"{name}.bucket.npy"
    cmd = [str(TOOL), "features", "--psv", str(psv_path), "--out", str(feat_path)]
    cmd += ["--bucket-out", str(bucket_path), "--threads", str(threads)]
    subprocess.run(cmd, check=True)
    feat = np.load(feat_path)
    rec = np.fromfile(psv_path, dtype=np.uint8).reshape(-1, RECORD)
    score = rec[:, 32:34].copy().view("<i2").ravel()
    ply = rec[:, 36:38].copy().view("<u2").ravel()
    result = rec[:, 38].copy().view(np.int8)
    used = np.abs(score.astype(np.int32)) < SCORE_DROP_ABS
    r = rarity(feat, counts)
    n_feat = (feat != np.iinfo(np.uint32).max).sum(axis=1)
    weight = np.ones(len(rec))
    if scan is not None:
        # ビン内一様抽出の重み = 母集団のビンの大きさ / 標本数 (Rust 側の min_count と一致することも確認する)
        b = log10_bin(r["min_count"])
        pop = np.array([x["population"] for x in scan["bins"]], dtype=np.float64)
        smp = np.array([x["sampled"] for x in scan["bins"]], dtype=np.float64)
        assert used.all(), "scan は |score| >= cap を除いているはず"
        assert np.array_equal(np.bincount(b, minlength=10), smp.astype(np.int64)), (
            "ビンの件数が scan と不一致"
        )
        weight = pop[b] / smp[b]
    np.savez(
        out_dir / f"{name}.meta.npz",
        score=score,
        result=result,
        ply=ply,
        used=used,
        design_weight=weight,
        **r,
    )

    mc = r["min_count"][used]
    info = {
        "name": name,
        "role": spec["role"],
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "files": [str(p) for p in paths],
        "records_per_file": recs,
        "population": total,
        "n": int(len(rec)),
        "sampling": sampling,
        "seed": spec["seed"],
        "counts": "full_counts.both.npy = sum of feature_counts/dlsuisho_unique_{001..030}.both.npy"
        " (loss 重み > 0 の record, 両視点)",
        "n_used_abs_score_lt_32000": int(used.sum()),
        "active_features_per_record": sorted(set(int(x) for x in np.unique(n_feat))),
        "min_count_zero_in_used": int((mc == 0).sum()),
        "records_per_log10_bin_used": {
            f"[1e{k},1e{k + 1})": int(v)
            for k, v in enumerate(np.bincount(log10_bin(mc), minlength=10))
        },
        "sha256": {
            f"{name}.psv": sha256_file(psv_path),
            f"{name}.feat.npy": sha256_file(feat_path),
            "full_counts.both.npy": sha256_file(counts_path),
            "loss_eval": sha256_file(TOOL),
        },
    }
    (out_dir / f"{name}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n")
    print(
        json.dumps(
            {k: v for k, v in info.items() if k not in ("files", "records_per_file")},
            ensure_ascii=False,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--n", type=int, default=1_000_000, help="一様標本の record 数")
    ap.add_argument("--only", choices=list(SETS))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts, counts_path = full_counts(args.out_dir)
    for name, spec in SETS.items():
        if args.only and name != args.only:
            continue
        make(name, spec, args.n, counts, counts_path, args.out_dir, args.threads)


if __name__ == "__main__":
    main()
