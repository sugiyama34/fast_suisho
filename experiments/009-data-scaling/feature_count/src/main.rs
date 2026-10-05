//! experiment-009: 教師 .psv 全体での HalfKA2 入力特徴の出現回数カウンタ。
//!
//! 各 .psv (PackedSfenValue 40 B/record) を mmap し、ファイルごとに
//! 131,949 次元の u64 カウントを出力する。arm (ファイル集合) のカウントは
//! 構成ファイルの出力を足すだけで得られる。
//!
//! 特徴抽出は BulletOu 2a8e5ed の `ShogiHalfKa2::map_features` をそのまま呼ぶ
//! (= 学習時の `PreparedData::new_with_pool` と同じ経路。
//!  crates/bulletou_lib/src/value/loader.rs の `inp.map_features_split(pos, ...)`)。
//!
//! 学習時のフィルタとの対応 (BulletOu 2a8e5ed の source study):
//! - .psv は `DirectSequentialDataLoader` で全 record を順に読む。record の読み飛ばしは無い
//! - `|score| >= --score-drop-abs` (既定 32000 = 詰みスタンプ) の record は削除されず、
//!   loss weight = 0 になる (loader.rs `if pos.score().unsigned_abs() >= cap { weight = 0.0 }`)。
//!   → 勾配に寄与しないので「学習が見る」カウントからは除外し、別配列 `dropped_both` に数える
//! - 玉が盤上に無い局面 (king sq = 81) は `map_halfka2_features` が特徴を 1 つも出さない。
//!   → `no_king` として数えるだけ (カウント配列には何も足さない)
//!
//! 出力 (ファイル stem ごと、--limit 指定時は stem に `.limitN` を付ける):
//! - `<stem>.stm.npy`          u64[131949]  学習対象 record の手番側視点の特徴
//! - `<stem>.nstm.npy`         u64[131949]  同 非手番側視点
//! - `<stem>.both.npy`         u64[131949]  stm + nstm
//! - `<stem>.dropped_both.npy` u64[131949]  |score| >= cap の record の stm + nstm
//! - `<stem>.json`             件数・検証統計のサマリ

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use bulletou_lib::game::inputs::{HALFKA2_DIMENSIONS, PIECE_INPUTS, ShogiHalfKa2, SparseInputType};
use bulletou_lib::shogi::{Color, PackedSfenValue, PieceType, ShogiBoard};
use clap::Parser;

/// PackedSfenValue のサイズ
const RECORD_SIZE: usize = 40;
/// 1 視点あたりの最大アクティブ特徴数 (盤上駒 + 持ち駒 = 40)
const MAX_ACTIVE: usize = 40;
/// HalfKA2 の自玉 plane の先頭 (F_KING = 1548)。敵玉 plane もここに畳まれる
const F_KING: usize = 1548;
/// 1 スレッドが一度に取る record 数
const CHUNK_RECORDS: usize = 1 << 20;

#[derive(Parser, Debug)]
#[command(about = "HalfKA2 feature occurrence counter for .psv teacher files (experiment-009)")]
struct Args {
    /// 入力 .psv ファイル (複数可、ファイルごとに別出力)
    #[arg(required = true)]
    files: Vec<PathBuf>,

    /// 出力ディレクトリ
    #[arg(long, default_value = "/mnt/nvme1/sugiyama/feature_counts")]
    out_dir: PathBuf,

    /// ワーカースレッド数
    #[arg(long, default_value_t = 16)]
    threads: usize,

    /// 各ファイルの先頭 N record だけ処理する (テスト用)
    #[arg(long)]
    limit: Option<usize>,

    /// BulletOu の --score-drop-abs と同じ意味 (0 で無効)。学習 run は既定値 32000
    #[arg(long, default_value_t = 32000)]
    score_drop_abs: u16,

    /// N record の SFEN と特徴インデックスを JSON Lines で stdout に出して終了する (検証用)
    #[arg(long)]
    dump: Option<usize>,

    /// --dump で出す record の間隔 (record index = k * stride)。ファイル全域から標本を取るため
    #[arg(long, default_value_t = 1)]
    dump_stride: usize,
}

/// スレッドごとのアキュムレータ
struct Acc {
    stm: Vec<u64>,
    nstm: Vec<u64>,
    dropped_both: Vec<u64>,
    positions: u64,
    trained: u64,
    dropped: u64,
    no_king: u64,
    /// 1 視点あたりのアクティブ特徴数のヒストグラム [0..=MAX_ACTIVE]
    active_hist: [u64; MAX_ACTIVE + 1],
    /// 検証違反: 自玉特徴が 1 個でない / king bucket が視点内で不一致 / 40 超 / 範囲外
    violations: u64,
}

impl Acc {
    fn new() -> Self {
        Self {
            stm: vec![0; HALFKA2_DIMENSIONS],
            nstm: vec![0; HALFKA2_DIMENSIONS],
            dropped_both: vec![0; HALFKA2_DIMENSIONS],
            positions: 0,
            trained: 0,
            dropped: 0,
            no_king: 0,
            active_hist: [0; MAX_ACTIVE + 1],
            violations: 0,
        }
    }

    fn merge(&mut self, o: &Acc) {
        for (a, b) in [(&mut self.stm, &o.stm), (&mut self.nstm, &o.nstm), (&mut self.dropped_both, &o.dropped_both)] {
            a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x += *y);
        }
        self.positions += o.positions;
        self.trained += o.trained;
        self.dropped += o.dropped;
        self.no_king += o.no_king;
        for i in 0..=MAX_ACTIVE {
            self.active_hist[i] += o.active_hist[i];
        }
        self.violations += o.violations;
    }
}

/// 1 視点の特徴リストの健全性検査。
/// 全特徴が同じ king bucket kb を持ち、自玉特徴 kb*1629 + 1548 + kb がちょうど 1 個あること。
fn check_perspective(feats: &[usize]) -> bool {
    if feats.is_empty() || feats.len() > MAX_ACTIVE {
        return false;
    }
    let kb = feats[0] / PIECE_INPUTS;
    let own_king = kb * PIECE_INPUTS + F_KING + kb;
    let mut own = 0;
    for &f in feats {
        if f >= HALFKA2_DIMENSIONS || f / PIECE_INPUTS != kb {
            return false;
        }
        own += (f == own_king) as usize;
    }
    own == 1
}

/// record を 1 つ処理してアキュムレータに足す
#[inline]
fn process_record(psv: &PackedSfenValue, cap: u16, acc: &mut Acc) {
    let mut stm = [0usize; MAX_ACTIVE + 8];
    let mut nstm = [0usize; MAX_ACTIVE + 8];
    let mut n = 0usize;
    ShogiHalfKa2.map_features(psv, |s, o| {
        // 40 超は違反として後で数える (配列外書き込みはしない)
        if n < stm.len() {
            stm[n] = s;
            nstm[n] = o;
        }
        n += 1;
    });

    acc.positions += 1;
    if n == 0 {
        acc.no_king += 1;
        return;
    }
    let ok = n <= MAX_ACTIVE && check_perspective(&stm[..n]) && check_perspective(&nstm[..n]);
    if !ok {
        acc.violations += 1;
        return;
    }
    acc.active_hist[n] += 1;

    if cap != 0 && psv.score().unsigned_abs() >= cap {
        acc.dropped += 1;
        for i in 0..n {
            acc.dropped_both[stm[i]] += 1;
            acc.dropped_both[nstm[i]] += 1;
        }
    } else {
        acc.trained += 1;
        for i in 0..n {
            acc.stm[stm[i]] += 1;
            acc.nstm[nstm[i]] += 1;
        }
    }
}

/// mmap した record 列 (&[u8]) を PackedSfenValue として読む
#[inline]
fn record_at(bytes: &[u8], i: usize) -> PackedSfenValue {
    let mut psv = PackedSfenValue::default();
    psv.as_bytes_mut().copy_from_slice(&bytes[i * RECORD_SIZE..(i + 1) * RECORD_SIZE]);
    psv
}

/// u64 配列を .npy (format v1.0, little-endian '<u8') で書く
fn write_npy_u64(path: &Path, data: &[u64]) -> std::io::Result<()> {
    let dict = format!("{{'descr': '<u8', 'fortran_order': False, 'shape': ({},), }}", data.len());
    // magic(6) + version(2) + header_len(2) + dict + padding + '\n' を 64 の倍数に揃える
    let unpadded = 10 + dict.len() + 1;
    let pad = (64 - unpadded % 64) % 64;
    let header = format!("{dict}{}\n", " ".repeat(pad));
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(b"\x93NUMPY\x01\x00")?;
    w.write_all(&(header.len() as u16).to_le_bytes())?;
    w.write_all(header.as_bytes())?;
    for &x in data {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()
}

// -----------------------------------------------------------------------------
// 検証用: ShogiBoard → SFEN 文字列 (cshogi との突き合わせ用。手数は 1 固定)
// -----------------------------------------------------------------------------

fn piece_char(pt: PieceType) -> &'static str {
    match pt {
        PieceType::Pawn => "P",
        PieceType::Lance => "L",
        PieceType::Knight => "N",
        PieceType::Silver => "S",
        PieceType::Bishop => "B",
        PieceType::Rook => "R",
        PieceType::Gold => "G",
        PieceType::King => "K",
        PieceType::ProPawn => "+P",
        PieceType::ProLance => "+L",
        PieceType::ProKnight => "+N",
        PieceType::ProSilver => "+S",
        PieceType::Horse => "+B",
        PieceType::Dragon => "+R",
        PieceType::None => "",
    }
}

fn board_to_sfen(b: &ShogiBoard) -> String {
    let mut s = String::new();
    // SFEN は 1 段目から、各段は 9 筋 → 1 筋の順。Square index = file*9 + rank
    for rank in 0..9 {
        let mut empty = 0;
        for file in (0..9).rev() {
            let p = b.board[file * 9 + rank];
            if p.is_none() {
                empty += 1;
                continue;
            }
            if empty > 0 {
                s += &empty.to_string();
                empty = 0;
            }
            let c = piece_char(p.piece_type);
            s += &if p.color == Color::Black { c.to_string() } else { c.to_lowercase() };
        }
        if empty > 0 {
            s += &empty.to_string();
        }
        if rank < 8 {
            s.push('/');
        }
    }
    s += if b.side_to_move == Color::Black { " b " } else { " w " };
    // 持ち駒は USI 慣例の R B G S N L P 順
    let order = [
        PieceType::Rook,
        PieceType::Bishop,
        PieceType::Gold,
        PieceType::Silver,
        PieceType::Knight,
        PieceType::Lance,
        PieceType::Pawn,
    ];
    let mut hand = String::new();
    for color in [Color::Black, Color::White] {
        for &pt in &order {
            let c = b.hand(color).count(pt);
            if c == 0 {
                continue;
            }
            if c > 1 {
                hand += &c.to_string();
            }
            let ch = piece_char(pt);
            hand += &if color == Color::Black { ch.to_string() } else { ch.to_lowercase() };
        }
    }
    s += if hand.is_empty() { "-" } else { &hand };
    s += " 1";
    s
}

fn dump(bytes: &[u8], n: usize, stride: usize) {
    let total = bytes.len() / RECORD_SIZE;
    for i in (0..total).step_by(stride.max(1)).take(n) {
        let psv = record_at(bytes, i);
        let board = ShogiBoard::from_packed_sfen(&psv);
        let mut stm = Vec::new();
        let mut nstm = Vec::new();
        ShogiHalfKa2.map_features(&psv, |s, o| {
            stm.push(s);
            nstm.push(o);
        });
        let line = serde_json::json!({
            "index": i,
            "sfen": board_to_sfen(&board),
            "score": psv.score(),
            "ply": psv.game_ply(),
            "stm": stm,
            "nstm": nstm,
        });
        println!("{line}");
    }
}

// -----------------------------------------------------------------------------

fn count_file(path: &Path, args: &Args) -> std::io::Result<()> {
    let file = File::open(path)?;
    // Safety: 入力は読み取り専用で、処理中に書き換えられない前提
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let _ = mmap.advise(memmap2::Advice::Sequential);
    let len = mmap.len();
    if len % RECORD_SIZE != 0 {
        eprintln!("warning: {} size {} is not a multiple of {RECORD_SIZE}", path.display(), len);
    }
    let total_records = len / RECORD_SIZE;
    let n_records = args.limit.map_or(total_records, |l| l.min(total_records));
    let bytes = &mmap[..n_records * RECORD_SIZE];

    if let Some(n) = args.dump {
        dump(bytes, n, args.dump_stride);
        return Ok(());
    }

    let cap = args.score_drop_abs;
    let threads = args.threads.max(1);
    let n_chunks = n_records.div_ceil(CHUNK_RECORDS);
    let next = AtomicUsize::new(0);
    let started = Instant::now();

    // 各スレッドが atomic カウンタでチャンクを取り合う (ほぼ先頭から順に読む)
    let accs: Vec<Acc> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut acc = Acc::new();
                    loop {
                        let c = next.fetch_add(1, Ordering::Relaxed);
                        if c >= n_chunks {
                            break;
                        }
                        let begin = c * CHUNK_RECORDS;
                        let end = (begin + CHUNK_RECORDS).min(n_records);
                        for i in begin..end {
                            process_record(&record_at(bytes, i), cap, &mut acc);
                        }
                    }
                    acc
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("worker panicked")).collect()
    });
    let mut acc = Acc::new();
    for a in &accs {
        acc.merge(a);
    }
    let elapsed = started.elapsed().as_secs_f64();

    // ---- 出力 ----
    std::fs::create_dir_all(&args.out_dir)?;
    let stem_base = path.file_stem().unwrap().to_string_lossy().to_string();
    let stem = match args.limit {
        Some(l) => format!("{stem_base}.limit{l}"),
        None => stem_base,
    };
    let both: Vec<u64> = acc.stm.iter().zip(&acc.nstm).map(|(a, b)| a + b).collect();
    write_npy_u64(&args.out_dir.join(format!("{stem}.stm.npy")), &acc.stm)?;
    write_npy_u64(&args.out_dir.join(format!("{stem}.nstm.npy")), &acc.nstm)?;
    write_npy_u64(&args.out_dir.join(format!("{stem}.both.npy")), &both)?;
    write_npy_u64(&args.out_dir.join(format!("{stem}.dropped_both.npy")), &acc.dropped_both)?;

    let zero_both = both.iter().filter(|&&x| x == 0).count();
    let feature_sum: u64 = both.iter().sum();
    let outputs: Vec<String> =
        ["stm", "nstm", "both", "dropped_both"].iter().map(|k| format!("{stem}.{k}.npy")).collect();
    let summary = serde_json::json!({
        "file": path.canonicalize()?.display().to_string(),
        "file_bytes": len,
        "records_in_file": total_records,
        "records_processed": n_records,
        "limit": args.limit,
        "positions": acc.positions,
        "trained_positions": acc.trained,
        "dropped_positions_score_cap": acc.dropped,
        "no_king_positions": acc.no_king,
        "violations": acc.violations,
        "score_drop_abs": cap,
        "feature_dims": HALFKA2_DIMENSIONS,
        "feature_sum_both": feature_sum,
        "zero_features_both": zero_both,
        "active_per_perspective_hist": acc.active_hist.to_vec(),
        "threads": threads,
        "elapsed_sec": elapsed,
        "records_per_sec": n_records as f64 / elapsed,
        "extractor": "bulletou_lib 2a8e5ed ShogiHalfKa2::map_features",
        "outputs": outputs,
    });
    std::fs::write(args.out_dir.join(format!("{stem}.json")), serde_json::to_string_pretty(&summary)? + "\n")?;
    eprintln!(
        "{}: {} records in {:.1}s ({:.2} M rec/s, {} threads); trained={} dropped={} no_king={} violations={} zero_features={}",
        path.display(),
        n_records,
        elapsed,
        n_records as f64 / elapsed / 1e6,
        threads,
        acc.trained,
        acc.dropped,
        acc.no_king,
        acc.violations,
        zero_both
    );
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    for f in &args.files {
        count_file(f, &args)?;
    }
    Ok(())
}
