use arrow::{
    array::StringArray,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use base64::Engine;
use parquet::arrow::ArrowWriter;
use std::{
    fs,
    process::Command,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn encode_writes_all_shards_and_tokenizer() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("nanochat-encode-{}-{nonce}", std::process::id()));
    let source = root.join("parquet");
    let output = root.join("tokens");
    fs::create_dir_all(&source).unwrap();
    fs::create_dir(&output).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
    let shards = [
        ("shard_00000.parquet", ["ab", "中文"]),
        ("shard_00002.parquet", ["second\nline", ""]),
        ("shard_99999.parquet", ["validation", "🙂"]),
    ];
    for (name, docs) in &shards {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(docs.to_vec()))],
        )
        .unwrap();
        let mut writer = ArrowWriter::try_new(
            fs::File::create(source.join(name)).unwrap(),
            schema.clone(),
            None,
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let vocab = root.join("vocab.tiktoken");
    let lines = (0u16..256)
        .map(|id| {
            format!(
                "{} {id}\n",
                base64::engine::general_purpose::STANDARD.encode([id as u8])
            )
        })
        .collect::<String>();
    fs::write(&vocab, lines).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_rust-bpe"))
        .arg("encode")
        .env("NANOCHAT_PARQUET_DIR", &source)
        .env("NANOCHAT_TOKEN_DIR", &output)
        .env("NANOCHAT_VOCAB_PATH", &vocab)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let mut names: Vec<_> = fs::read_dir(&output)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "shard_00000.bin",
            "shard_00001.bin",
            "shard_00002.bin",
            "vocab.tiktoken"
        ]
    );
    for (index, (_, docs)) in shards.into_iter().enumerate() {
        let data = fs::read(output.join(format!("shard_{index:05}.bin"))).unwrap();
        let tokens: Vec<u16> = data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(tokens[0], 32759);
        assert_eq!(tokens.iter().filter(|&&id| id == 32759).count(), 2);
        let expected: Vec<u16> = docs
            .into_iter()
            .flat_map(|doc| std::iter::once(32759).chain(doc.bytes().map(u16::from)))
            .collect();
        assert_eq!(tokens, expected);
    }
    assert_eq!(
        fs::read(output.join("vocab.tiktoken")).unwrap(),
        fs::read(&vocab).unwrap()
    );
    assert!(!output.with_extension("part").exists());

    // 已完成的数据目录不能被下一次编码覆盖。
    let repeated = Command::new(env!("CARGO_BIN_EXE_rust-bpe"))
        .arg("encode")
        .env("NANOCHAT_PARQUET_DIR", &source)
        .env("NANOCHAT_TOKEN_DIR", &output)
        .env("NANOCHAT_VOCAB_PATH", &vocab)
        .output()
        .unwrap();
    assert!(!repeated.status.success());
    assert!(String::from_utf8_lossy(&repeated.stderr).contains("token directory is not empty"));

    // 后面的分片读取失败时，不发布前面已编码的部分结果。
    fs::remove_dir_all(&output).unwrap();
    fs::write(source.join("shard_99999.parquet"), b"invalid parquet").unwrap();
    let failed = Command::new(env!("CARGO_BIN_EXE_rust-bpe"))
        .arg("encode")
        .env("NANOCHAT_PARQUET_DIR", &source)
        .env("NANOCHAT_TOKEN_DIR", &output)
        .env("NANOCHAT_VOCAB_PATH", &vocab)
        .output()
        .unwrap();
    assert!(!failed.status.success());
    assert!(!output.exists());
    assert!(output.with_extension("part").is_dir());

    // 修复源数据后重试，自动清理上次留下的临时目录。
    let stale = output.with_extension("part").join("stale/nested");
    fs::create_dir_all(&stale).unwrap();
    fs::write(stale.join("old.bin"), b"old tokens").unwrap();
    fs::copy(
        source.join("shard_00000.parquet"),
        source.join("shard_99999.parquet"),
    )
    .unwrap();
    let retried = Command::new(env!("CARGO_BIN_EXE_rust-bpe"))
        .arg("encode")
        .env("NANOCHAT_PARQUET_DIR", &source)
        .env("NANOCHAT_TOKEN_DIR", &output)
        .env("NANOCHAT_VOCAB_PATH", &vocab)
        .output()
        .unwrap();
    assert!(
        retried.status.success(),
        "{}",
        String::from_utf8_lossy(&retried.stderr)
    );
    assert!(!output.with_extension("part").exists());
    assert!(!output.join("stale").exists());
    assert_eq!(
        fs::read(output.join("shard_00002.bin")).unwrap(),
        fs::read(output.join("shard_00000.bin")).unwrap()
    );
    fs::remove_dir_all(root).unwrap();
}
