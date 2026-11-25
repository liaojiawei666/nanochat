use crate::bpe::{BpeEncoder, TokenId};
use crate::dataset::Dataset;
use crate::{paths, special};
use rayon::prelude::*;
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};

fn write_shard_bin(encoder: &BpeEncoder, out_dir: &Path, shard_idx: usize, docs: &[String]) {
    assert!(!docs.is_empty(), "shard {shard_idx} contains no documents");
    let encoded: Vec<Vec<TokenId>> = docs.par_iter().map(|doc| encoder.encode(doc)).collect();
    let name = format!("shard_{shard_idx:05}.bin");
    let path = out_dir.join(&name);
    let temporary = out_dir.join(format!("{name}.part"));
    let mut out = BufWriter::new(File::create(&temporary).expect("create token shard"));
    for ids in &encoded {
        // 文档边界由数据格式决定，BPE 编码器只处理正文。
        for id in std::iter::once(special::SpecialToken::Bos.id()).chain(ids.iter().copied()) {
            assert!(id <= u16::MAX as u32, "token id {id} does not fit in u16");
            let bytes = (id as u16).to_le_bytes();
            out.write_all(&bytes).expect("write tokens");
        }
    }
    out.flush().expect("flush token shard");
    out.get_ref().sync_all().expect("sync token shard");
    fs::rename(temporary, &path).expect("publish token shard");
    eprintln!("wrote {} docs -> {}", docs.len(), path.display());
}

pub fn run() {
    let vocab_path = paths::vocab_tiktoken_path();
    let vocab = fs::read(&vocab_path).expect("read tokenizer");
    let encoder = BpeEncoder::from_tiktoken_path(&vocab_path);
    let out_dir = paths::token_bin_dir();
    let all_paths = paths::list_all_shard_paths();
    assert!(all_paths.len() >= 2, "need at least two parquet shards");
    assert!(
        !out_dir.exists()
            || fs::read_dir(&out_dir)
                .expect("read token directory")
                .next()
                .is_none(),
        "token directory is not empty; choose a new NANOCHAT_TOKEN_DIR"
    );
    // 全部分片和词表先写入相邻临时目录，完成后一次性发布。
    let mut temporary_name = out_dir
        .file_name()
        .expect("token directory needs a name")
        .to_os_string();
    temporary_name.push(".part");
    let stage = out_dir.with_file_name(temporary_name);
    fs::create_dir_all(out_dir.parent().unwrap()).expect("create token directory parent");
    if stage.exists() {
        fs::remove_dir_all(&stage).expect("remove stale temporary token directory");
    }
    fs::create_dir(&stage).expect("create temporary token directory");
    for (index, path) in all_paths.iter().enumerate() {
        let docs: Vec<String> = Dataset::new(vec![path.clone()]).collect();
        write_shard_bin(&encoder, &stage, index, &docs);
    }
    assert_eq!(
        vocab,
        fs::read(vocab_path).expect("read tokenizer"),
        "tokenizer changed while encoding"
    );
    let mut file = File::create(stage.join("vocab.tiktoken")).expect("create tokenizer");
    file.write_all(&vocab).expect("write tokenizer");
    file.sync_all().expect("sync tokenizer");
    if out_dir.exists() {
        fs::remove_dir(&out_dir).expect("remove empty token directory");
    }
    fs::rename(stage, out_dir).expect("publish token directory");
}
