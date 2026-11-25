use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

use rayon::prelude::*;

use crate::bpe::{BpeEncoder, BpeTrainer};
use crate::dataset::Dataset;
use crate::paths;
use crate::special::VOCAB_NO_SPECIAL_SIZE;

/// 用验证集正文 token（不含 BOS）测压缩率，写到 `data/tokenizer/val_compress.txt`。
fn eval_val_compress(encoder: &BpeEncoder, shard_list: Vec<PathBuf>) {
    let (docs, bytes, tokens) = Dataset::new(shard_list)
        .par_bridge()
        .map(|text| {
            let b = text.len() as u64;
            let t = encoder.encode(&text).len() as u64;
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
    let line =
        format!("climbmix-val | docs={docs} bytes={bytes} tokens={tokens} bytes/token={ratio:.4}");
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

/// 按 Unicode 字符截断；累计超过预算的那篇文档仍保留，随后停止读取。
fn training_documents(
    mut documents: impl Iterator<Item = String> + Send,
    doc_cap: usize,
    max_chars: u64,
) -> impl Iterator<Item = String> + Send {
    let mut chars_seen = 0u64;
    std::iter::from_fn(move || {
        if chars_seen > max_chars {
            return None;
        }
        let mut text = documents.next()?;
        if let Some((end, _)) = text.char_indices().nth(doc_cap) {
            text.truncate(end);
        }
        chars_seen += text.chars().count() as u64;
        Some(text)
    })
}

pub fn run() {
    let mut train_shards = paths::list_all_shard_paths();
    assert!(
        train_shards.len() >= 2,
        "need at least two parquet shards for train/validation isolation"
    );
    // 排序后的最后一个 shard 用于验证，其余用于训练。
    let val_shard = train_shards.pop().unwrap();
    let doc_cap = 10_000;
    let max_chars = 2_000_000_000;
    eprintln!("tok train: doc_cap={doc_cap} max_chars={max_chars}");
    let documents = training_documents(Dataset::new(train_shards), doc_cap, max_chars);
    let mut trainer = BpeTrainer::from_documents(documents);
    trainer.train(VOCAB_NO_SPECIAL_SIZE);

    let vocab_path = paths::vocab_tiktoken_path();
    trainer.write_vocab(&vocab_path);

    let encoder = BpeEncoder::from_tiktoken_path(&vocab_path);
    eval_val_compress(&encoder, vec![val_shard]);
}

#[cfg(test)]
mod tests {
    use super::training_documents;

    #[test]
    fn limits_unicode_documents_without_reading_past_budget() {
        let documents = vec!["你好🙂abc".to_string(), "é文xyz".to_string()]
            .into_iter()
            .chain(std::iter::once_with(|| {
                panic!("read past the character budget")
            }));
        let texts: Vec<_> = training_documents(documents, 4, 7).collect();
        assert_eq!(texts, ["你好🙂a", "é文xy"]);
    }

    #[test]
    fn stops_only_after_exceeding_budget() {
        let documents = ["ab", "cd", "ef", "gh"].into_iter().map(String::from);
        let texts: Vec<_> = training_documents(documents, 10, 4).collect();
        assert_eq!(texts, ["ab", "cd", "ef"]);
    }
}
