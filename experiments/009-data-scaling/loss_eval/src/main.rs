//! experiment-009: 学習済み SFNN (HalfKA2 1024-7-64 k3k3) の順伝播を CPU で行い、
//! 固定した教師標本の上で loss を測るためのツール。
//!
//! サブコマンド:
//! - `features` : .psv の各 record の HalfKA2 特徴 (両視点 40 個ずつ) と layer stack 番号を .npy に書く
//! - `eval`     : ネットの出力を .npy に書き、BulletOu の validation と同じ関数で loss / accuracy を表示する
//!
//! 順伝播は 2 種類:
//! - 量子化 (`--net nn.bin`): やねうら王 SFNN (`SFNN_halfka2_1024-7-64-k3k3.h`) の整数演算をそのまま再現する。
//!   出力は `fc_2` + shortcut の生の i32 (スケール QA*QB = 127*64 = 8128)。
//!   エンジンの評価値は `raw / FV_SCALE` (C++ の整数除算) を ±VALUE_MAX_EVAL でクリップしたもの
//! - float (`--state state.bin`): checkpoint の畳み込み前の f32 重みを BulletOu の CPU validation と同じ手順で
//!   畳み込み (`cuda_cpp_sfnn_weights_for_cpu_validation`)、`fast_sfnn.rs` の scalar forward で計算する。
//!   出力は logit (sigmoid をかける前の値)
//!
//! 特徴抽出は `ShogiHalfKa2::map_features`、layer stack は `ShogiKingRankBucket::<9>` (学習・validation と同じ関数)。
//! 入力局面は `.psv` (40 B/record) か、`--test-teacher` (BulletOu の `read_random_teacher_positions` で
//! 学習時の validation と同じ標本を取る)。

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use bulletou_lib::game::inputs::{HALFKA2_DIMENSIONS, PIECE_INPUTS, ShogiHalfKa2, SparseInputType};
use bulletou_lib::game::outputs::{OutputBuckets, ShogiKingRankBucket};
use bulletou_lib::shogi::PackedSfenValue;
use bulletou_lib::validate::{
    AccuracyReport, ValidationLossKind, compute_sign_accuracy_with_loss, read_random_teacher_positions,
};
use clap::{Parser, Subcommand};

const RECORD_SIZE: usize = 40;
const MAX_ACTIVE: usize = 40;
const FT: usize = 1024;
const HALF: usize = FT / 2;
const L1_OUT: usize = 8; // 7 hidden + 1 shortcut
const L1_HIDDEN: usize = 7;
const L2_IN: usize = 14;
const L2_IN_PAD: usize = 32;
const L2: usize = 64;
const STACKS: usize = 9;
const N_VIRTUAL: usize = PIECE_INPUTS; // w_factor の行数
const SFNN_VERSION: u32 = 0x7AF3_2F16;
const LEB128_MAGIC: &[u8] = b"COMPRESSED_LEB128";
/// 量子化出力 → logit のスケール (QA * QB)
const OUTPUT_SCALE: f32 = 127.0 * 64.0;

#[derive(Parser, Debug)]
#[command(about = "SFNN HalfKA2 1024-7-64 k3k3 CPU forward / loss tool (experiment-009)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// .psv の特徴 index (u32 [N, 80] = 手番側 40 + 非手番側 40, 不足は u32::MAX) と stack 番号 (u8 [N]) を書く
    Features {
        #[arg(long)]
        psv: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        bucket_out: Option<PathBuf>,
        #[arg(long, default_value_t = 8)]
        threads: usize,
    },
    /// 教師ファイル全体を走査し、record ごとの min_count (80 特徴の出現回数の最小値) の log10 ビンごとに
    /// 一様な部分標本を選ぶ (bottom-k: record 番号のハッシュが小さい K 個)。|score| >= cap の record は除く
    Scan {
        /// 教師 .psv (連結した record 番号で index を返す)
        #[arg(long, value_delimiter = ',', required = true)]
        files: Vec<PathBuf>,
        /// 出現回数 (.npy, u64 [131949])
        #[arg(long)]
        counts: PathBuf,
        /// ビンごとの標本数の上限
        #[arg(long, default_value_t = 100_000)]
        per_bin: usize,
        #[arg(long)]
        seed: u64,
        #[arg(long, default_value_t = 32000)]
        score_drop_abs: u16,
        /// 出力: <prefix>.index.npy (u64, 昇順), <prefix>.scan.json (ビンごとの母集団の record 数など)
        #[arg(long)]
        out_prefix: PathBuf,
        #[arg(long, default_value_t = 8)]
        threads: usize,
    },
    /// ネットの出力を計算する
    Eval {
        /// 量子化ネット (やねうら王が読む nn.bin)
        #[arg(long)]
        net: Option<PathBuf>,
        /// checkpoint の state.bin (float, 畳み込み前)
        #[arg(long)]
        state: Option<PathBuf>,
        /// (診断用) nn.bin の重みを逆量子化して float forward でも計算する (量子化誤差の内訳: 重みの丸め / 活性の丸め)
        #[arg(long, requires = "net")]
        dequant: bool,
        /// 入力 .psv
        #[arg(long, conflicts_with = "test_teacher")]
        psv: Option<PathBuf>,
        /// BulletOu の --test-teacher と同じ標本を使う
        #[arg(long)]
        test_teacher: Option<String>,
        #[arg(long, default_value_t = 100_000)]
        test_positions: usize,
        #[arg(long, default_value_t = 0)]
        test_seed: u64,
        /// --test-teacher の標本を .psv として書き出す
        #[arg(long)]
        dump_psv: Option<PathBuf>,
        /// 先頭 N record だけ (テスト用)
        #[arg(long)]
        limit: Option<usize>,
        /// 出力 prefix: <prefix>.q.npy (i32 生出力) / <prefix>.f.npy (f32 logit)
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value_t = 8)]
        threads: usize,
        /// 表示する loss の設定 (BulletOu の --lambda / --scale / --score-drop-abs)
        #[arg(long, default_value_t = 1.0)]
        lambda: f32,
        #[arg(long, default_value_t = 290.0)]
        scale: f32,
        #[arg(long, default_value_t = 32000)]
        score_drop_abs: u16,
    },
}

// ============================================================================ npy

fn write_npy(path: &Path, descr: &str, shape: &[usize], data: &[u8]) -> std::io::Result<()> {
    let shape_s = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_s}, }}");
    let unpadded = 10 + dict.len() + 1;
    let pad = (64 - unpadded % 64) % 64;
    let header = format!("{dict}{}\n", " ".repeat(pad));
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(b"\x93NUMPY\x01\x00")?;
    w.write_all(&(header.len() as u16).to_le_bytes())?;
    w.write_all(header.as_bytes())?;
    w.write_all(data)?;
    w.flush()
}

fn suffixed(prefix: &Path, ext: &str) -> PathBuf {
    PathBuf::from(format!("{}.{ext}", prefix.display()))
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // Safety: plain-old-data の数値配列だけに使う
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

// ============================================================================ 入力局面

enum Positions {
    Mmap(memmap2::Mmap, usize),
    Vec(Vec<PackedSfenValue>),
}

impl Positions {
    fn len(&self) -> usize {
        match self {
            Positions::Mmap(_, n) => *n,
            Positions::Vec(v) => v.len(),
        }
    }
    fn get(&self, i: usize) -> PackedSfenValue {
        match self {
            Positions::Mmap(m, _) => {
                let mut psv = PackedSfenValue::default();
                psv.as_bytes_mut().copy_from_slice(&m[i * RECORD_SIZE..(i + 1) * RECORD_SIZE]);
                psv
            }
            Positions::Vec(v) => v[i],
        }
    }
}

fn open_psv(path: &Path, limit: Option<usize>) -> std::io::Result<Positions> {
    let f = File::open(path)?;
    // Safety: 読み取り専用。処理中に書き換えられない前提
    let m = unsafe { memmap2::Mmap::map(&f)? };
    if m.len() % RECORD_SIZE != 0 {
        return Err(std::io::Error::other(format!("{}: size is not a multiple of 40", path.display())));
    }
    let n = limit.map_or(m.len() / RECORD_SIZE, |l| l.min(m.len() / RECORD_SIZE));
    Ok(Positions::Mmap(m, n))
}

/// 1 局面の特徴 (両視点) と stack 番号。玉が無い局面は None
struct Feats {
    stm: [u32; MAX_ACTIVE],
    nstm: [u32; MAX_ACTIVE],
    n: usize,
    bucket: usize,
}

fn extract(psv: &PackedSfenValue) -> Option<Feats> {
    let mut f = Feats { stm: [u32::MAX; MAX_ACTIVE], nstm: [u32::MAX; MAX_ACTIVE], n: 0, bucket: 0 };
    let mut n = 0usize;
    let mut over = false;
    ShogiHalfKa2.map_features(psv, |s, o| {
        if n < MAX_ACTIVE {
            f.stm[n] = s as u32;
            f.nstm[n] = o as u32;
        } else {
            over = true;
        }
        n += 1;
    });
    if over {
        panic!("more than {MAX_ACTIVE} active features");
    }
    if n == 0 {
        return None;
    }
    f.n = n;
    f.bucket = ShogiKingRankBucket::<9>.bucket(psv);
    Some(f)
}

// ============================================================================ nn.bin (量子化)

struct QStack {
    l1b: [i32; L1_OUT],
    l1w: Vec<i8>, // [8][1024]
    l2b: [i32; L2],
    l2w: Vec<i8>, // [64][32]
    l3b: i32,
    l3w: [i8; L2],
}

struct QNet {
    ft_b: Vec<i16>,
    ft_w: Vec<i16>, // [131949][1024]
    stacks: Vec<QStack>,
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        s
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn leb128_i16(&mut self, count: usize) -> Vec<i16> {
        assert_eq!(self.take(LEB128_MAGIC.len()), LEB128_MAGIC, "LEB128 magic");
        let size = self.u32() as usize;
        let payload = self.take(size);
        let mut out = Vec::with_capacity(count);
        let mut i = 0usize;
        while i < payload.len() {
            let mut result: i32 = 0;
            let mut shift = 0u32;
            loop {
                let byte = payload[i];
                i += 1;
                result |= i32::from(byte & 0x7F) << shift;
                shift += 7;
                if byte & 0x80 == 0 {
                    if shift < 32 && byte & 0x40 != 0 {
                        result |= !0i32 << shift;
                    }
                    break;
                }
            }
            out.push(i16::try_from(result).expect("LEB128 value out of i16 range"));
        }
        assert_eq!(out.len(), count, "LEB128 value count");
        out
    }
}

fn load_qnet(path: &Path) -> std::io::Result<QNet> {
    let raw = std::fs::read(path)?;
    let mut r = Reader { b: &raw, p: 0 };
    let version = r.u32();
    assert_eq!(version, SFNN_VERSION, "not an SFNN nn.bin");
    let _hash = r.u32();
    let dlen = r.u32() as usize;
    let desc = String::from_utf8_lossy(r.take(dlen)).to_string();
    // 説明文字列は学習器ごとに違う (水匠 11 は "HalfKA(Friend)") ので、次元と stack 数だけ確かめる。
    // 実際の形はこの後の読み取りでバイト数が合うこと (末尾まで過不足なし) で確かめる
    assert!(
        desc.contains(&format!("[{HALFKA2_DIMENSIONS}->{FT}x2]")) && desc.contains("LayerStack=9"),
        "arch: {desc}"
    );
    let _ft_hash = r.u32();
    let ft_b = r.leb128_i16(FT);
    let ft_w = r.leb128_i16(HALFKA2_DIMENSIONS * FT);
    let mut stacks = Vec::with_capacity(STACKS);
    for _ in 0..STACKS {
        let _net_hash = r.u32();
        let mut l1b = [0i32; L1_OUT];
        l1b.iter_mut().for_each(|x| *x = r.i32());
        let l1w: Vec<i8> = r.take(L1_OUT * FT).iter().map(|&b| b as i8).collect();
        let mut l2b = [0i32; L2];
        l2b.iter_mut().for_each(|x| *x = r.i32());
        let l2w: Vec<i8> = r.take(L2 * L2_IN_PAD).iter().map(|&b| b as i8).collect();
        let l3b = r.i32();
        let mut l3w = [0i8; L2];
        l3w.iter_mut().zip(r.take(L2)).for_each(|(x, &b)| *x = b as i8);
        stacks.push(QStack { l1b, l1w, l2b, l2w, l3b, l3w });
    }
    assert_eq!(r.p, raw.len(), "trailing bytes in nn.bin");
    eprintln!("loaded {} ({desc})", path.display());
    Ok(QNet { ft_b, ft_w, stacks })
}

/// やねうら王 SFNN の整数順伝播。戻り値: (fc_2 + shortcut の生出力, i16 アキュムレータのオーバーフロー有無)
fn forward_q(net: &QNet, f: &Feats, acc: &mut [i32; FT], x: &mut [u8; FT]) -> (i32, bool) {
    let mut overflow = false;
    for (p, feats) in [&f.stm[..f.n], &f.nstm[..f.n]].into_iter().enumerate() {
        for (a, &b) in acc.iter_mut().zip(&net.ft_b) {
            *a = i32::from(b);
        }
        for &fi in feats {
            let row = &net.ft_w[fi as usize * FT..fi as usize * FT + FT];
            for (a, &w) in acc.iter_mut().zip(row) {
                *a += i32::from(w);
            }
        }
        // エンジンは読み込み時に重み・バイアスを 2 倍し、i16 で累積する (nnue_feature_transformer.h scale_weights)
        let out = &mut x[p * HALF..(p + 1) * HALF];
        for j in 0..HALF {
            let s0 = 2 * acc[j];
            let s1 = 2 * acc[j + HALF];
            overflow |= s0 != i32::from(s0 as i16) || s1 != i32::from(s1 as i16);
            let a0 = i32::from(s0 as i16).clamp(0, 254);
            let a1 = i32::from(s1 as i16).min(254);
            // SIMD 経路: mulhi(a0 << 7, a1) = floor(a0*a1/512)、負は packus で 0
            let prod = a0 * a1;
            out[j] = if prod <= 0 { 0 } else { (prod >> 9) as u8 };
        }
    }
    let st = &net.stacks[f.bucket];
    // fc_0 (1024 -> 8)
    let mut y0 = [0i32; L1_OUT];
    for o in 0..L1_OUT {
        let w = &st.l1w[o * FT..(o + 1) * FT];
        let mut s = 0i32;
        for (&wi, &xi) in w.iter().zip(x.iter()) {
            s += i32::from(wi) * i32::from(xi);
        }
        y0[o] = st.l1b[o] + s;
    }
    // [SqrClippedReLU(0..7), ClippedReLU(0..7)]
    let mut in1 = [0i32; L2_IN];
    for o in 0..L1_HIDDEN {
        let v = i64::from(y0[o]);
        in1[o] = ((v * v) >> 19).min(127) as i32;
        in1[L1_HIDDEN + o] = (y0[o] >> 6).clamp(0, 127);
    }
    // fc_1 (14 -> 64) + ClippedReLU
    let mut c1 = [0i32; L2];
    for k in 0..L2 {
        let w = &st.l2w[k * L2_IN_PAD..k * L2_IN_PAD + L2_IN];
        let mut s = st.l2b[k];
        for i in 0..L2_IN {
            s += i32::from(w[i]) * in1[i];
        }
        c1[k] = (s >> 6).clamp(0, 127);
    }
    // fc_2 (64 -> 1) + shortcut
    let mut y2 = st.l3b;
    for i in 0..L2 {
        y2 += i32::from(st.l3w[i]) * c1[i];
    }
    (y2 + y0[L1_HIDDEN], overflow)
}

// ============================================================================ state.bin (float)

struct FNet {
    l0w: Vec<f32>, // 畳み込み済み [131949][1024]
    l0b: Vec<f32>,
    l1w: Vec<f32>, // [9][8][1024]
    l1b: Vec<f32>,
    l2w: Vec<f32>, // [9][64][14]
    l2b: Vec<f32>,
    l3w: Vec<f32>, // [9][64]
    l3b: Vec<f32>,
}

fn state_records(path: &Path) -> std::io::Result<Vec<(String, u64, usize)>> {
    let mut f = File::open(path)?;
    let size = f.metadata()?.len();
    let mut off = 0u64;
    let mut recs = Vec::new();
    while off < size {
        f.seek(SeekFrom::Start(off))?;
        let mut head = vec![0u8; 512.min((size - off) as usize)];
        f.read_exact(&mut head)?;
        let nl = head.iter().position(|&b| b == b'\n').expect("record id newline");
        let id = String::from_utf8_lossy(&head[..nl]).to_string();
        let n = u64::from_le_bytes(head[nl + 1..nl + 9].try_into().unwrap()) as usize;
        let val_off = off + nl as u64 + 9;
        recs.push((id, val_off, n));
        off = val_off + 4 * n as u64;
    }
    assert_eq!(off, size, "state.bin trailing bytes");
    Ok(recs)
}

fn load_f32(path: &Path, recs: &[(String, u64, usize)], id: &str, expect: usize) -> std::io::Result<Vec<f32>> {
    let (_, off, n) = recs.iter().find(|r| r.0 == id).unwrap_or_else(|| panic!("{id} not in state.bin"));
    assert_eq!(*n, expect, "{id} length");
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(*off))?;
    let mut buf = vec![0u8; n * 4];
    f.read_exact(&mut buf)?;
    Ok(buf.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// BulletOu `cuda_cpp_sfnn_weights_for_cpu_validation` (factorizer = shared, 既定) と同じ畳み込み
fn load_fnet(path: &Path) -> std::io::Result<FNet> {
    let recs = state_records(path)?;
    let raw = load_f32(path, &recs, "nnue/weights/l0w", (HALFKA2_DIMENSIONS + N_VIRTUAL) * FT)?;
    // fold_sfnn_halfka2_piece_factorized_l0w: 行 r = w_specific[r] + w_factor[r % 1629] (f32 加算)
    let mut l0w = vec![0f32; HALFKA2_DIMENSIONS * FT];
    for r in 0..HALFKA2_DIMENSIONS {
        let v = HALFKA2_DIMENSIONS + r % N_VIRTUAL;
        for j in 0..FT {
            l0w[r * FT + j] = raw[r * FT + j] + raw[v * FT + j];
        }
    }
    drop(raw);
    let l0b = load_f32(path, &recs, "nnue/weights/l0b", FT)?;
    let mut l1w = load_f32(path, &recs, "nnue/weights/l1w", STACKS * L1_OUT * FT)?;
    let mut l1b = load_f32(path, &recs, "nnue/weights/l1b", STACKS * L1_OUT)?;
    let mut l2w = load_f32(path, &recs, "nnue/weights/l2w", STACKS * L2 * L2_IN)?;
    let mut l2b = load_f32(path, &recs, "nnue/weights/l2b", STACKS * L2)?;
    let mut l3w = load_f32(path, &recs, "nnue/weights/l3w", STACKS * L2)?;
    let mut l3b = load_f32(path, &recs, "nnue/weights/l3b", STACKS)?;
    let has = |id: &str| recs.iter().any(|r| r.0 == id);
    if has("nnue/weights/l1fw") {
        // fold_cuda_cpp_sfnn_l1f_into_stacked_l1: l1fw は入力優先 [1024][8]
        let fw = load_f32(path, &recs, "nnue/weights/l1fw", FT * L1_OUT)?;
        let fb = load_f32(path, &recs, "nnue/weights/l1fb", L1_OUT)?;
        for s in 0..STACKS {
            for o in 0..L1_OUT {
                l1b[s * L1_OUT + o] += fb[o];
                for i in 0..FT {
                    l1w[s * L1_OUT * FT + o * FT + i] += fw[i * L1_OUT + o];
                }
            }
        }
    }
    if has("nnue/weights/l2fw") {
        let fw = load_f32(path, &recs, "nnue/weights/l2fw", L2 * L2_IN)?;
        let fb = load_f32(path, &recs, "nnue/weights/l2fb", L2)?;
        for s in 0..STACKS {
            for o in 0..L2 {
                l2b[s * L2 + o] += fb[o];
                for i in 0..L2_IN {
                    l2w[s * L2 * L2_IN + o * L2_IN + i] += fw[o * L2_IN + i];
                }
            }
        }
    }
    if has("nnue/weights/l3fw") {
        let fw = load_f32(path, &recs, "nnue/weights/l3fw", L2)?;
        let fb = load_f32(path, &recs, "nnue/weights/l3fb", 1)?;
        for s in 0..STACKS {
            l3b[s] += fb[0];
            for i in 0..L2 {
                l3w[s * L2 + i] += fw[i];
            }
        }
    }
    for id in ["l1axw", "l2axw", "l3axw"] {
        assert!(!has(&format!("nnue/weights/{id}")), "axis factorizer is not supported");
    }
    eprintln!("loaded {} (float, factorizers folded)", path.display());
    Ok(FNet { l0w, l0b, l1w, l1b, l2w, l2b, l3w, l3b })
}

/// nn.bin の重みを逆量子化した FNet (FT は /127, 全結合の重みは /64, バイアスは /(127*64))
fn dequant_fnet(q: &QNet) -> FNet {
    let bs = OUTPUT_SCALE;
    let mut f = FNet {
        l0w: q.ft_w.iter().map(|&v| f32::from(v) / 127.0).collect(),
        l0b: q.ft_b.iter().map(|&v| f32::from(v) / 127.0).collect(),
        l1w: Vec::new(),
        l1b: Vec::new(),
        l2w: Vec::new(),
        l2b: Vec::new(),
        l3w: Vec::new(),
        l3b: Vec::new(),
    };
    for st in &q.stacks {
        f.l1w.extend(st.l1w.iter().map(|&v| f32::from(v) / 64.0));
        f.l1b.extend(st.l1b.iter().map(|&v| v as f32 / bs));
        for k in 0..L2 {
            f.l2w.extend(st.l2w[k * L2_IN_PAD..k * L2_IN_PAD + L2_IN].iter().map(|&v| f32::from(v) / 64.0));
        }
        f.l2b.extend(st.l2b.iter().map(|&v| v as f32 / bs));
        f.l3w.extend(st.l3w.iter().map(|&v| f32::from(v) / 64.0));
        f.l3b.push(st.l3b as f32 / bs);
    }
    f
}

/// `fast_sfnn.rs` の scalar forward と同じ計算順序
fn forward_f(net: &FNet, f: &Feats, acc: &mut [[f32; FT]; 2], comb: &mut [f32; FT]) -> f32 {
    const SCALE: f32 = 127.0 / 128.0;
    for (p, feats) in [&f.stm[..f.n], &f.nstm[..f.n]].into_iter().enumerate() {
        let a = &mut acc[p];
        a.copy_from_slice(&net.l0b);
        for &fi in feats {
            let row = &net.l0w[fi as usize * FT..fi as usize * FT + FT];
            for (x, &w) in a.iter_mut().zip(row) {
                *x += w;
            }
        }
        for x in a.iter_mut() {
            *x = x.clamp(0.0, 1.0);
        }
        for i in 0..HALF {
            comb[p * HALF + i] = a[i] * a[HALF + i] * SCALE;
        }
    }
    let s = f.bucket;
    let mut l1 = [0f32; L1_OUT];
    for o in 0..L1_OUT {
        let w = &net.l1w[s * L1_OUT * FT + o * FT..s * L1_OUT * FT + (o + 1) * FT];
        let mut v = net.l1b[s * L1_OUT + o];
        for i in 0..FT {
            if comb[i] != 0.0 {
                v += w[i] * comb[i];
            }
        }
        l1[o] = v;
    }
    let skip = l1[L1_HIDDEN];
    let mut in2 = [0f32; L2_IN];
    for r in 0..L1_HIDDEN {
        in2[r] = (l1[r].abs() * l1[r].abs() * SCALE).clamp(0.0, 1.0);
        in2[L1_HIDDEN + r] = l1[r].clamp(0.0, 1.0);
    }
    let mut l2 = [0f32; L2];
    for o in 0..L2 {
        let mut v = net.l2b[s * L2 + o];
        for i in 0..L2_IN {
            if in2[i] != 0.0 {
                v += net.l2w[s * L2 * L2_IN + o * L2_IN + i] * in2[i];
            }
        }
        l2[o] = v.clamp(0.0, 1.0);
    }
    let mut out = net.l3b[s];
    for i in 0..L2 {
        if l2[i] != 0.0 {
            out += net.l3w[s * L2 + i] * l2[i];
        }
    }
    out + skip
}

// ============================================================================ scan (稀さの層ごとの標本)

const N_BINS: usize = 10; // log10(min_count) = 0..9

fn read_npy_u64(path: &Path) -> std::io::Result<Vec<u64>> {
    let b = std::fs::read(path)?;
    assert_eq!(&b[..8], b"\x93NUMPY\x01\x00", "npy v1.0 only");
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let header = String::from_utf8_lossy(&b[10..10 + hl]);
    assert!(header.contains("'descr': '<u8'"), "{}: dtype must be <u8", path.display());
    Ok(b[10 + hl..].chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect())
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn log10_bin(m: u64) -> usize {
    let mut b = 0usize;
    let mut t = 10u64;
    while m >= t && b + 1 < N_BINS {
        b += 1;
        t = t.saturating_mul(10);
    }
    b
}

#[derive(Default)]
struct ScanAcc {
    heaps: Vec<std::collections::BinaryHeap<(u64, u64)>>,
    pop: [u64; N_BINS],
    zero: u64,
    dropped: u64,
    no_king: u64,
}

#[allow(clippy::too_many_arguments)]
fn scan(
    files: &[PathBuf],
    counts: &[u64],
    per_bin: usize,
    seed: u64,
    cap: u16,
    threads: usize,
) -> std::io::Result<(Vec<Vec<(u64, u64)>>, ScanAcc, u64)> {
    const CHUNK: usize = 1 << 20;
    let mut total = ScanAcc { heaps: vec![Default::default(); N_BINS], ..Default::default() };
    let mut base = 0u64;
    for path in files {
        let t0 = Instant::now();
        let f = File::open(path)?;
        // Safety: 読み取り専用
        let m = unsafe { memmap2::Mmap::map(&f)? };
        let _ = m.advise(memmap2::Advice::Sequential);
        let n = m.len() / RECORD_SIZE;
        let n_chunks = n.div_ceil(CHUNK);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let accs: Vec<ScanAcc> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads.max(1))
                .map(|_| {
                    sc.spawn(|| {
                        let mut acc = ScanAcc { heaps: vec![Default::default(); N_BINS], ..Default::default() };
                        loop {
                            let c = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if c >= n_chunks {
                                break;
                            }
                            for i in c * CHUNK..((c + 1) * CHUNK).min(n) {
                                let mut psv = PackedSfenValue::default();
                                psv.as_bytes_mut().copy_from_slice(&m[i * RECORD_SIZE..(i + 1) * RECORD_SIZE]);
                                if cap != 0 && psv.score().unsigned_abs() >= cap {
                                    acc.dropped += 1;
                                    continue;
                                }
                                let mut mc = u64::MAX;
                                let mut nf = 0usize;
                                ShogiHalfKa2.map_features(&psv, |s, o| {
                                    mc = mc.min(counts[s]).min(counts[o]);
                                    nf += 1;
                                });
                                if nf == 0 {
                                    acc.no_king += 1;
                                    continue;
                                }
                                if mc == 0 {
                                    acc.zero += 1;
                                    continue;
                                }
                                let b = log10_bin(mc);
                                acc.pop[b] += 1;
                                let g = base + i as u64;
                                let key = splitmix64(seed ^ splitmix64(g));
                                let h = &mut acc.heaps[b];
                                if h.len() < per_bin {
                                    h.push((key, g));
                                } else if key < h.peek().unwrap().0 {
                                    h.pop();
                                    h.push((key, g));
                                }
                            }
                        }
                        acc
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("worker panicked")).collect()
        });
        for a in accs {
            for b in 0..N_BINS {
                total.pop[b] += a.pop[b];
                total.heaps[b].extend(a.heaps[b].iter().copied());
                while total.heaps[b].len() > per_bin {
                    total.heaps[b].pop();
                }
            }
            total.zero += a.zero;
            total.dropped += a.dropped;
            total.no_king += a.no_king;
        }
        eprintln!("scan {}: {n} records in {:.1}s", path.display(), t0.elapsed().as_secs_f64());
        base += n as u64;
    }
    let picked = total.heaps.iter().map(|h| h.iter().copied().collect()).collect();
    Ok((picked, total, base))
}

// ============================================================================ 並列実行

/// 局面を threads 個の連続区間に分けて f を呼ぶ (出力スライスは区間ごと)
fn par_chunks<T: Send, S: Send + Default, F>(n: usize, out: &mut [T], threads: usize, f: F) -> Vec<S>
where
    F: Fn(usize, &mut [T], &mut S) + Sync,
{
    if n == 0 {
        return Vec::new();
    }
    let per = n.div_ceil(threads.max(1)).max(1);
    let stride = out.len() / n.max(1);
    std::thread::scope(|sc| {
        let hs: Vec<_> = out
            .chunks_mut(per * stride)
            .enumerate()
            .map(|(k, chunk)| {
                let f = &f;
                sc.spawn(move || {
                    let mut st = S::default();
                    f(k * per, chunk, &mut st);
                    st
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("worker panicked")).collect()
    })
}

#[derive(Default)]
struct Stat {
    no_king: usize,
    overflow: usize,
}

fn report(label: &str, outputs: &[f32], pos: &Positions, lambda: f32, scale: f32, cap: u16) -> AccuracyReport {
    let n = pos.len();
    let scores: Vec<i16> = (0..n).map(|i| pos.get(i).score()).collect();
    let results: Vec<i8> = (0..n).map(|i| pos.get(i).game_result()).collect();
    let cap = if cap > 0 { Some(cap) } else { None };
    let r = compute_sign_accuracy_with_loss(
        outputs,
        &scores,
        &results,
        cap,
        lambda,
        scale,
        1.0,
        ValidationLossKind::SigmoidMse,
    );
    println!(
        "{label}: accuracy={:.7} ({}/{} decisive; draws={}; mate filtered={}) loss={:.8} (n={})  [bulletou_lib::validate::compute_sign_accuracy_with_loss]",
        r.accuracy(),
        r.sign_matches,
        r.compared,
        r.drawn_games,
        r.filtered_by_score_cap,
        r.test_loss.unwrap_or(f32::NAN),
        r.loss_sampled
    );
    r
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Scan { files, counts, per_bin, seed, score_drop_abs, out_prefix, threads } => {
            let counts = read_npy_u64(&counts)?;
            assert_eq!(counts.len(), HALFKA2_DIMENSIONS);
            let t0 = Instant::now();
            let (picked, acc, total) = scan(&files, &counts, per_bin, seed, score_drop_abs, threads)?;
            let mut idx: Vec<u64> = picked.iter().flatten().map(|&(_, g)| g).collect();
            idx.sort_unstable();
            write_npy(&suffixed(&out_prefix, "index.npy"), "<u8", &[idx.len()], as_bytes(&idx))?;
            let bins: Vec<serde_json::Value> = (0..N_BINS)
                .map(|b| {
                    serde_json::json!({
                        "bin": b, "lo": 10u64.pow(b as u32), "population": acc.pop[b], "sampled": picked[b].len()
                    })
                })
                .collect();
            let info = serde_json::json!({
                "files": files.iter().map(|f| f.display().to_string()).collect::<Vec<_>>(),
                "records_total": total,
                "dropped_score_cap": acc.dropped,
                "no_king": acc.no_king,
                "min_count_zero": acc.zero,
                "per_bin": per_bin,
                "seed": seed,
                "selection": "bin = floor(log10(min_count)); per bin the per_bin records with the smallest splitmix64(seed ^ splitmix64(global_index))",
                "bins": bins,
                "threads": threads,
                "elapsed_sec": t0.elapsed().as_secs_f64(),
            });
            std::fs::write(suffixed(&out_prefix, "scan.json"), serde_json::to_string_pretty(&info)? + "\n")?;
            eprintln!("{}", serde_json::to_string(&info)?);
        }
        Cmd::Features { psv, out, bucket_out, threads } => {
            let pos = open_psv(&psv, None)?;
            let n = pos.len();
            let t0 = Instant::now();
            // 1 行 = 特徴 80 個 + stack 番号 (玉が無い局面は特徴 u32::MAX, stack 255)
            let mut rows = vec![([u32::MAX; 2 * MAX_ACTIVE], 255u8); n];
            let stats: Vec<Stat> = par_chunks(n, &mut rows, threads, |start, chunk, st: &mut Stat| {
                for (k, row) in chunk.iter_mut().enumerate() {
                    match extract(&pos.get(start + k)) {
                        Some(f) => {
                            row.0[..MAX_ACTIVE].copy_from_slice(&f.stm);
                            row.0[MAX_ACTIVE..].copy_from_slice(&f.nstm);
                            row.1 = f.bucket as u8;
                        }
                        None => st.no_king += 1,
                    }
                }
            });
            let feats: Vec<u32> = rows.iter().flat_map(|r| r.0).collect();
            let buckets: Vec<u8> = rows.iter().map(|r| r.1).collect();
            write_npy(&out, "<u4", &[n, 2 * MAX_ACTIVE], as_bytes(&feats))?;
            if let Some(b) = bucket_out {
                write_npy(&b, "|u1", &[n], &buckets)?;
            }
            let no_king: usize = stats.iter().map(|s| s.no_king).sum();
            eprintln!("features: {n} records, no_king={no_king}, {:.1}s", t0.elapsed().as_secs_f64());
        }
        Cmd::Eval {
            net,
            state,
            dequant,
            psv,
            test_teacher,
            test_positions,
            test_seed,
            dump_psv,
            limit,
            out,
            threads,
            lambda,
            scale,
            score_drop_abs,
        } => {
            assert!(net.is_some() || state.is_some(), "--net または --state を指定");
            let pos = if let Some(p) = &psv {
                open_psv(p, limit)?
            } else if let Some(t) = &test_teacher {
                let t0 = Instant::now();
                let v = read_random_teacher_positions(t, test_positions, test_seed)?;
                eprintln!(
                    "sampled {} positions from {t} (n={test_positions}, seed={test_seed}) in {:.1}s",
                    v.len(),
                    t0.elapsed().as_secs_f64()
                );
                let v = match limit {
                    Some(l) => v.into_iter().take(l).collect(),
                    None => v,
                };
                if let Some(d) = &dump_psv {
                    let mut w = BufWriter::new(File::create(d)?);
                    for p in &v {
                        w.write_all(p.as_bytes())?;
                    }
                    w.flush()?;
                }
                Positions::Vec(v)
            } else {
                panic!("--psv または --test-teacher を指定");
            };
            let n = pos.len();
            let mut logits_q: Option<Vec<f32>> = None;
            if let Some(netp) = &net {
                let t0 = Instant::now();
                let q = load_qnet(netp)?;
                let tl = t0.elapsed().as_secs_f64();
                let t1 = Instant::now();
                let mut raw = vec![0i32; n];
                let stats: Vec<Stat> = par_chunks(n, &mut raw, threads, |start, chunk, st: &mut Stat| {
                    let mut acc = [0i32; FT];
                    let mut x = [0u8; FT];
                    for (k, o) in chunk.iter_mut().enumerate() {
                        match extract(&pos.get(start + k)) {
                            Some(f) => {
                                let (v, ov) = forward_q(&q, &f, &mut acc, &mut x);
                                st.overflow += ov as usize;
                                *o = v;
                            }
                            None => {
                                st.no_king += 1;
                                *o = 0;
                            }
                        }
                    }
                });
                let te = t1.elapsed().as_secs_f64();
                let no_king: usize = stats.iter().map(|s| s.no_king).sum();
                let overflow: usize = stats.iter().map(|s| s.overflow).sum();
                eprintln!(
                    "quantized forward: {n} positions, load {tl:.1}s, forward {te:.1}s ({:.0} pos/s, {threads} threads), no_king={no_king}, i16_overflow={overflow}",
                    n as f64 / te
                );
                if let Some(o) = &out {
                    write_npy(&suffixed(o, "q.npy"), "<i4", &[n], as_bytes(&raw))?;
                }
                let lg: Vec<f32> = raw.iter().map(|&v| v as f32 / OUTPUT_SCALE).collect();
                report("quantized (nn.bin)", &lg, &pos, lambda, scale, score_drop_abs);
                if dequant {
                    let fq = dequant_fnet(&q);
                    let mut dl = vec![0f32; n];
                    par_chunks(n, &mut dl, threads, |start, chunk, _: &mut Stat| {
                        let mut acc = [[0f32; FT]; 2];
                        let mut comb = [0f32; FT];
                        for (k, o) in chunk.iter_mut().enumerate() {
                            if let Some(f) = extract(&pos.get(start + k)) {
                                *o = forward_f(&fq, &f, &mut acc, &mut comb);
                            }
                        }
                    });
                    if let Some(o) = &out {
                        write_npy(&suffixed(o, "dq.npy"), "<f4", &[n], as_bytes(&dl))?;
                    }
                    report("dequantized weights, float activations", &dl, &pos, lambda, scale, score_drop_abs);
                }
                logits_q = Some(lg);
            }
            if let Some(sp) = &state {
                let t0 = Instant::now();
                let fnet = load_fnet(sp)?;
                let tl = t0.elapsed().as_secs_f64();
                let t1 = Instant::now();
                let mut lg = vec![0f32; n];
                par_chunks(n, &mut lg, threads, |start, chunk, st: &mut Stat| {
                    let mut acc = [[0f32; FT]; 2];
                    let mut comb = [0f32; FT];
                    for (k, o) in chunk.iter_mut().enumerate() {
                        match extract(&pos.get(start + k)) {
                            Some(f) => *o = forward_f(&fnet, &f, &mut acc, &mut comb),
                            None => {
                                st.no_king += 1;
                                *o = 0.0;
                            }
                        }
                    }
                });
                let te = t1.elapsed().as_secs_f64();
                eprintln!(
                    "float forward: {n} positions, load {tl:.1}s, forward {te:.1}s ({:.0} pos/s)",
                    n as f64 / te
                );
                if let Some(o) = &out {
                    write_npy(&suffixed(o, "f.npy"), "<f4", &[n], as_bytes(&lg))?;
                }
                report("float (state.bin)", &lg, &pos, lambda, scale, score_drop_abs);
                if let Some(q) = &logits_q {
                    let (mut sum, mut max) = (0f64, 0f64);
                    for (a, b) in q.iter().zip(&lg) {
                        let d = f64::from((a - b).abs());
                        sum += d;
                        max = max.max(d);
                    }
                    println!(
                        "quantized vs float logit: mean |diff|={:.6} ({:.2} cp at scale {scale}), max |diff|={:.6}",
                        sum / n as f64,
                        sum / n as f64 * f64::from(scale),
                        max
                    );
                }
            }
        }
    }
    Ok(())
}
