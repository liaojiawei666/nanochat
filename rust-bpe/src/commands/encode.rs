use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use rayon::prelude::*;

use crate::bpe::{BpeEncoder, TokenId};
use crate::dataset::{Dataset, DatasetOptions, Split};
use crate::paths;

fn write_shard_bin(encoder: &BpeEncoder, out_dir: &Path, shard_idx: usize, docs: &[String]) {
    if docs.is_empty() {
        return;
    }

    let encoded: Vec<Vec<TokenId>> = docs.par_iter().map(|doc| encoder.encode(doc)).collect();

    let path = out_dir.join(format!("shard_{shard_idx:05}.bin"));
    let mut out = BufWriter::new(File::create(&path).unwrap_or_else(|e| {
        panic!("create {}: {e}", path.display());
    }));
    for ids in &encoded {
        for &id in ids {
            assert!(id <= u16::MAX as u32, "token id {id} does not fit in u16");
            out.write_all(&(id as u16).to_le_bytes())
                .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        }
    }
    out.flush()
        .unwrap_or_else(|e| panic!("flush {}: {e}", path.display()));
    eprintln!("wrote {} docs -> {}", docs.len(), path.display());
}

pub fn run() {
    let encoder = BpeEncoder::from_tiktoken_path(&paths::vocab_tiktoken_path());

    let out_dir = paths::token_bin_dir();
    std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| {
        panic!("create dir {}: {e}", out_dir.display());
    });

    let mut cur_shard: Option<usize> = None;
    let mut docs: Vec<String> = Vec::new();

    let finish_shard = |shard_idx: usize, docs: &mut Vec<String>| {
        write_shard_bin(&encoder, &out_dir, shard_idx, docs);
        docs.clear();
    };

    for sample in Dataset::new(DatasetOptions::new(Split::Train)) {
        if cur_shard != Some(sample.shard_idx) {
            if let Some(prev) = cur_shard {
                finish_shard(prev, &mut docs);
            }
            cur_shard = Some(sample.shard_idx);
        }
        assert_eq!(
            sample.doc_idx,
            docs.len(),
            "shard {}: doc order broken at doc_idx={}",
            sample.shard_idx,
            sample.doc_idx
        );
        docs.push(sample.text);
    }
    if let Some(shard_idx) = cur_shard {
        finish_shard(shard_idx, &mut docs);
    }
}
