use std::path::PathBuf;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data")
}

/// 预训练 parquet：`data/pretrain/climbmix/parquet`
pub fn parquet_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("NANOCHAT_PARQUET_DIR") {
        return PathBuf::from(path);
    }
    data_dir().join("pretrain/climbmix/parquet")
}

/// 列出目录中的 parquet 分片并排序；训练/验证划分由命令决定。
pub fn list_all_shard_paths() -> Vec<PathBuf> {
    let dir = parquet_dir();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read dir {}: {e}", dir.display()))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            if name.ends_with(".parquet") && !name.ends_with(".tmp") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no parquet shards in {}", dir.display());
    paths
}

/// 编码后的 token bin：`data/pretrain/climbmix/tokenized`
pub fn token_bin_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("NANOCHAT_TOKEN_DIR") {
        return PathBuf::from(path);
    }
    data_dir().join("pretrain/climbmix/tokenized")
}

/// 词表文件：`data/tokenizer/vocab.tiktoken`
pub fn vocab_tiktoken_path() -> PathBuf {
    if let Some(path) = std::env::var_os("NANOCHAT_VOCAB_PATH") {
        return PathBuf::from(path);
    }
    data_dir().join("tokenizer/vocab.tiktoken")
}

/// 验证集压缩率：`data/tokenizer/val_compress.txt`
pub fn val_compress_path() -> PathBuf {
    data_dir().join("tokenizer/val_compress.txt")
}
