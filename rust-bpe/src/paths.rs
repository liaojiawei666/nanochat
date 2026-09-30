use std::path::PathBuf;

fn home_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME is not set");
    PathBuf::from(home)
}

fn local_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../local")
}

/// 预训练 parquet：`~/.cache/nanochat/base_data_climbmix`
pub fn parquet_dir() -> PathBuf {
    home_dir().join(".cache/nanochat/base_data_climbmix")
}

/// 编码后的 token bin：`~/.cache/nanochat/tokenized_climbmix`
pub fn token_bin_dir() -> PathBuf {
    home_dir().join(".cache/nanochat/tokenized_climbmix")
}

/// 词表文件：`local/vocab.tiktoken`
pub fn vocab_tiktoken_path() -> PathBuf {
    local_dir().join("vocab.tiktoken")
}

/// 验证集压缩率：`local/val_compress.txt`
pub fn val_compress_path() -> PathBuf {
    local_dir().join("val_compress.txt")
}
