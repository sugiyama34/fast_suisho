"""ネット (nn.bin) の loss を固定標本の上で測り、稀さで層別して JSON に書く (experiment-009)。

順伝播は Rust の `loss_eval eval` (やねうら王 SFNN と同一の整数演算, 検証は notes/loss-eval.md)。
標本は make_samples.py が作ったもの (既定 A_s1 = S1 の train loss, B_s0 = S0 の held-out)。

loss の定義 (BulletOu 2a8e5ed の validation = `test_value_loss` と同じ):
    logit  = raw / 8128                       (raw = fc_2 + shortcut の生 i32, 8128 = QA*QB)
    target = lambda * sigmoid(score / scale) + (1 - lambda) * result01   (既定 lambda=1, scale=290)
    loss   = mean over |score| < 32000 of (sigmoid(logit) - target)^2
MAE は同じ record 上の |scale * logit - score| (cp, 教師のスケール)。
層別:
    by_rarest_log10 : 80 個の特徴のうち full 教師での出現回数が最小のもの (min_count) の log10 ビン
    by_n_lt_1e4     : full 教師での出現回数が 10^4 未満の特徴の個数 (両視点 80 個中)

使い方:
    data/matchenv/bin/python experiments/009-data-scaling/loss_eval/loss_eval.py \\
        --net data/bulletou/checkpoints/009-full/0012/nn.bin
    # 出力: /mnt/nvme1/sugiyama/loss_eval/<label>/loss.json と <set>.q.npy (record ごとの生出力)
    # 2 つのネットの対応のある差 (同じ record 上の loss の差) を層ごとに:
    data/matchenv/bin/python experiments/009-data-scaling/loss_eval/loss_eval.py \\
        --diff /mnt/nvme1/sugiyama/loss_eval/<label_a> /mnt/nvme1/sugiyama/loss_eval/<label_b>
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TOOL = HERE / "target" / "release" / "loss_eval"
SAMPLES = Path("/mnt/nvme1/sugiyama/loss_samples")
OUT_ROOT = Path("/mnt/nvme1/sugiyama/loss_eval")
CKPT_ROOT = REPO / "data" / "bulletou" / "checkpoints"
ABLATED_ROOT = Path("/mnt/nvme1/sugiyama/ablated")
OUTPUT_SCALE = 127 * 64
SCORE_DROP_ABS = 32000
LOG10_BINS = list(range(0, 10))  # [10^k, 10^(k+1))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def default_label(net: Path) -> str:
    net = net.resolve()
    parent = net.parent
    if parent.parent.parent == CKPT_ROOT.resolve():
        return f"{parent.parent.name}-{parent.name}"  # 例: 009-full-0012
    if parent.parent == ABLATED_ROOT:
        return f"ablated-{parent.name}"
    return parent.name


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def per_record(raw: np.ndarray, meta, lam: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """record ごとの (loss, 誤差 cp)。float64 で計算する。"""
    logit = raw.astype(np.float64) / OUTPUT_SCALE
    score = meta["score"].astype(np.float64)
    r = meta["result"].astype(np.int64)
    res01 = np.where(r > 0, 1.0, np.where(r < 0, 0.0, 0.5))
    target = lam * sigmoid(score / scale) + (1.0 - lam) * res01
    loss = (sigmoid(logit) - target) ** 2
    err = scale * logit - score
    return loss, err


def summarize(loss: np.ndarray, err: np.ndarray, w: np.ndarray) -> dict:
    """重み付き平均 (一様標本は w = 1)。loss_se = sqrt(sum w^2 (l - mean)^2) / sum w。"""
    n = int(loss.size)
    if n == 0:
        return {"n": 0}
    sw = w.sum()
    mean = float((w * loss).sum() / sw)
    d = {"n": n, "loss": mean, "mae_cp": float((w * np.abs(err)).sum() / sw)}
    d["loss_se"] = float(np.sqrt((w**2 * (loss - mean) ** 2).sum()) / sw) if n > 1 else None
    if not np.all(w == w[0]):
        d["n_eff"] = float(sw**2 / (w**2).sum())
    return d


def weights(meta) -> np.ndarray:
    return meta["design_weight"] if "design_weight" in meta else np.ones(len(meta["score"]))


def strata(meta) -> dict[str, list[tuple[str, np.ndarray]]]:
    """層の定義: {層別の名前: [(ラベル, bool マスク), ...]}"""
    out = {"by_rarest_log10": [], "by_n_lt_1e4": []}
    thr = np.array([10**k for k in range(1, 10)], dtype=np.uint64)
    lg = np.searchsorted(thr, meta["min_count"].astype(np.uint64), side="right")  # floor(log10), 整数比較
    for k in LOG10_BINS:
        out["by_rarest_log10"].append((f"[1e{k},1e{k + 1})", lg == k))
    nl = meta["n_lt_1e4"].astype(np.int64)
    for v in np.unique(nl):
        out["by_n_lt_1e4"].append((str(int(v)), nl == v))
    return out


def eval_set(net: Path, name: str, out_dir: Path, args) -> dict:
    psv = args.samples_dir / f"{name}.psv"
    meta = np.load(args.samples_dir / f"{name}.meta.npz")
    cmd = [str(TOOL), "eval", "--net", str(net), "--psv", str(psv), "--out", str(out_dir / name)]
    cmd += ["--threads", str(args.threads), "--lambda", str(args.lam), "--scale", str(args.scale)]
    cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
    raw = np.load(out_dir / f"{name}.q.npy")
    used = meta["used"]
    w = weights(meta)
    loss, err = per_record(raw, meta, args.lam, args.scale)
    m = re.search(r"loss=([0-9.]+) \(n=(\d+)\)", cp.stdout)
    t = re.search(r"forward ([0-9.]+)s", cp.stderr)
    d = {
        "sample": name,
        "n_records": int(raw.size),
        "design_weighted": bool(not np.all(w == 1)),
        "overall": summarize(loss[used], err[used], w[used]),
        "bulletou_validate_loss_f32_unweighted": float(m.group(1)) if m else None,
        "forward_sec": float(t.group(1)) if t else None,
    }
    for key, groups in strata(meta).items():
        d[key] = [
            {"bin": lab, **summarize(loss[used & mk], err[used & mk], w[used & mk])}
            for lab, mk in groups
            if (used & mk).any()
        ]
    return d


def default_sets(samples_dir: Path) -> list[str]:
    return [n for n in ("A_s1", "B_s0", "C_s1_rare", "D_s0_rare") if (samples_dir / f"{n}.meta.npz").exists()]


def run(args) -> None:
    net = args.net.resolve()
    label = args.label or default_label(net)
    out_dir = args.out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {
        "label": label,
        "net": str(net),
        "net_sha256": sha256_file(net),
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool_sha256": sha256_file(TOOL),
        "definition": {
            "forward": "YaneuraOu SFNN_halfka2_1024-7-64-k3k3 integer forward (exact); logit = raw / 8128",
            "loss": "mean over |score| < 32000 of (sigmoid(logit) - target)^2 (BulletOu validation definition)",
            "target": f"lambda*sigmoid(score/{args.scale}) + (1-lambda)*result01, lambda={args.lam}",
            "mae_cp": f"mean |{args.scale}*logit - score| over the same records",
            "strata_counts": "full teacher counts (sum of dlsuisho_unique_{001..030}.both.npy)",
            "weights": "A_s1/B_s0 are uniform (w=1); C_s1_rare/D_s0_rare are stratified by the log10 bin of "
            "min_count, w = population of the bin / sampled, so weighted means estimate the population",
        },
        "samples": {},
    }
    for name in args.sets.split(",") if args.sets else default_sets(args.samples_dir):
        res["samples"][name] = eval_set(net, name, out_dir, args)
        o = res["samples"][name]["overall"]
        print(f"{label} {name}: n={o['n']} loss={o['loss']:.6f} mae={o['mae_cp']:.1f}cp")
    (out_dir / "loss.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {out_dir / 'loss.json'}")


def diff(a_dir: Path, b_dir: Path, args) -> None:
    """同じ record 上での loss の差 (a - b) を層ごとに。SE は対応のある差の標準誤差。"""
    ja = json.loads((a_dir / "loss.json").read_text())
    jb = json.loads((b_dir / "loss.json").read_text())
    res = {"a": ja["label"], "b": jb["label"], "samples": {}}
    for name in ja["samples"]:
        if name not in jb["samples"]:
            continue
        meta = np.load(args.samples_dir / f"{name}.meta.npz")
        used = meta["used"]
        w = weights(meta)
        ra, rb = np.load(a_dir / f"{name}.q.npy"), np.load(b_dir / f"{name}.q.npy")
        la, _ = per_record(ra, meta, args.lam, args.scale)
        lb, _ = per_record(rb, meta, args.lam, args.scale)
        d = la - lb
        changed = ra != rb

        def row(m: np.ndarray) -> dict:
            e = summarize(d[m], d[m], w[m])
            return {"n": e["n"], "dloss": e["loss"], "dloss_se": e["loss_se"], "n_changed_output": int(changed[m].sum())}

        s = {"overall": row(used)}
        for key, groups in strata(meta).items():
            s[key] = [{"bin": lab, **row(used & mk)} for lab, mk in groups if (used & mk).any()]
        res["samples"][name] = s
    out = json.dumps(res, ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(out + "\n")
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", type=Path, help="nn.bin")
    ap.add_argument(
        "--diff", type=Path, nargs=2, metavar=("A_DIR", "B_DIR"), help="2 つの出力の対応のある差"
    )
    ap.add_argument("--label", help="出力ディレクトリ名 (既定: checkpoint なら <run>-<NNNN>)")
    ap.add_argument("--sets", help="カンマ区切り (既定: samples-dir にある A_s1,B_s0,C_s1_rare,D_s0_rare)")
    ap.add_argument("--out", type=Path, help="--diff の結果を書く JSON")
    ap.add_argument("--samples-dir", type=Path, default=SAMPLES)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=290.0)
    args = ap.parse_args()
    if args.diff:
        diff(args.diff[0], args.diff[1], args)
    elif args.net:
        run(args)
    else:
        ap.error("--net か --diff を指定")


if __name__ == "__main__":
    main()
