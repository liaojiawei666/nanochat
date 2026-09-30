use std::{
    fs::File,
    io::{BufWriter, Write},
};

use fancy_regex::Regex;
use rayon::prelude::*;

use crate::bpe::{count_pretok_strs, BpeEncoder, BpeTrainer};
use crate::dataset::{Dataset, DatasetOptions, Split};
use crate::paths;
use crate::special::{GPT4_PATTERN, VOCAB_NO_SPECIAL_SIZE};

/// 在验证集上测压缩率（bytes / tokens），写到 `local/val_compress.txt`。
fn eval_val_compress(encoder: &BpeEncoder) {
    let (docs, bytes, tokens) = Dataset::new(DatasetOptions::new(Split::Val))
        .par_bridge()
        .map(|sample| {
            let b = sample.text.len() as u64;
            let t = encoder.encode(&sample.text).len() as u64;
            (1u64, b, t)
        })
        .reduce(
            || (0u64, 0u64, 0u64),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );

    let ratio = if tokens == 0 {
        0.0
    } else {
        bytes as f64 / tokens as f64
    };
    let line = format!(
        "climbmix-val | docs={docs} bytes={bytes} tokens={tokens} bytes/token={ratio:.4}"
    );
    eprintln!("{line}");

    let path = paths::val_compress_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("create {}: {e}", parent.display()));
    }
    let mut out = BufWriter::new(File::create(&path).unwrap_or_else(|e| {
        panic!("create {}: {e}", path.display());
    }));
    writeln!(out, "{line}").unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    out.flush()
        .unwrap_or_else(|e| panic!("flush {}: {e}", path.display()));
    eprintln!("wrote {}", path.display());
}

pub fn run() {
    let pattern = Regex::new(GPT4_PATTERN).expect("invalid GPT4 regex");
    // 与 nanochat tok_train 一致：Train + doc_cap=10k + max_chars=2e9。
    let options = DatasetOptions::new(Split::Train)
        .doc_cap(10_000)
        .max_chars(2_000_000_000);
    eprintln!(
        "tok train: split=Train doc_cap={:?} max_chars={:?}",
        options.doc_cap, options.max_chars
    );
    let str_count = count_pretok_strs(options, &pattern);
    let mut trainer = BpeTrainer::from_str_counts(&str_count);
    trainer.train(VOCAB_NO_SPECIAL_SIZE);

    let vocab_path = paths::vocab_tiktoken_path();
    trainer.write_vocab(&vocab_path);

    let encoder = BpeEncoder::from_tiktoken_path(&vocab_path);
    eval_val_compress(&encoder);
}
