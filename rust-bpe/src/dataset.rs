use arrow::array::{Array, AsArray, StringArrayType};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use std::fs::File;
use std::path::PathBuf;

use crate::paths;

/// 和 nanochat 一致：最后一个 parquet 是验证集，其余是训练集。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// 除最后一个外的全部 shard。
    Train,
    /// 仅最后一个 shard（测压缩率等）。
    Val,
}

/// 一篇文档及其在数据集中的位置。
pub struct DocSample {
    pub text: String,
    pub doc_idx: usize,
    /// 在全部排序后的 parquet 列表中的下标（验证集仍是最后一个下标）。
    pub shard_idx: usize,
}

/// 数据集读取选项。默认是读完整个 split，不截断、不限量。
#[derive(Clone, Copy, Debug)]
pub struct DatasetOptions {
    pub split: Split,
    /// 每篇文档最多保留的字符数（Unicode scalar）；`None` 表示不截断。
    pub doc_cap: Option<usize>,
    /// 累计产出字符数上限；超过后停止。`None` 表示不限制。
    pub max_chars: Option<u64>,
}

impl DatasetOptions {
    /// 默认：读完整个 split。
    pub fn new(split: Split) -> Self {
        Self {
            split,
            doc_cap: None,
            max_chars: None,
        }
    }

    pub fn doc_cap(mut self, doc_cap: usize) -> Self {
        self.doc_cap = Some(doc_cap);
        self
    }

    pub fn max_chars(mut self, max_chars: u64) -> Self {
        self.max_chars = Some(max_chars);
        self
    }
}

fn list_all_shard_paths() -> Vec<PathBuf> {
    let dir = paths::parquet_dir();
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
    if paths.is_empty() {
        panic!("no parquet shards in {}", dir.display());
    }
    paths
}

/// `(全局 shard 下标, 路径)`，按 split 筛选。
fn list_shard_paths(split: Split) -> Vec<(usize, PathBuf)> {
    let all = list_all_shard_paths();
    let last = all.len() - 1;
    match split {
        Split::Train => all
            .into_iter()
            .enumerate()
            .filter(|(i, _)| *i != last)
            .collect(),
        Split::Val => vec![(last, all[last].clone())],
    }
}

fn value_at<'a>(arr: &impl StringArrayType<'a>, row: usize) -> Option<&'a str> {
    if arr.is_null(row) {
        None
    } else {
        Some(arr.value(row))
    }
}

fn doc_at(batch: &RecordBatch, row: usize) -> Option<&str> {
    let col = batch
        .column_by_name("text")
        .expect("parquet shard missing column 'text'");
    match col.data_type() {
        DataType::Utf8 => value_at(&col.as_string::<i32>(), row),
        DataType::LargeUtf8 => value_at(&col.as_string::<i64>(), row),
        DataType::Utf8View => value_at(&col.as_string_view(), row),
        other => panic!("unexpected text column type: {other}"),
    }
}

/// 按字符数截断，与 Python `s[:n]` 一致。
fn truncate_chars(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        None => s,
        Some((byte_idx, _)) => &s[..byte_idx],
    }
}

/// 按文档迭代的数据集：每次 `next` 产出一篇 `DocSample`。
pub struct Dataset {
    paths: Vec<(usize, PathBuf)>,
    path_idx: usize,
    shard_idx: usize,
    doc_idx: usize,
    reader: Option<ParquetRecordBatchReader>,
    batch: Option<RecordBatch>,
    row: usize,
    doc_cap: Option<usize>,
    max_chars: Option<u64>,
    chars_seen: u64,
    exhausted: bool,
}

impl Dataset {
    pub fn new(options: DatasetOptions) -> Self {
        Self {
            paths: list_shard_paths(options.split),
            path_idx: 0,
            shard_idx: 0,
            doc_idx: 0,
            reader: None,
            batch: None,
            row: 0,
            doc_cap: options.doc_cap,
            max_chars: options.max_chars,
            chars_seen: 0,
            exhausted: false,
        }
    }

    fn open_reader(path: &PathBuf) -> ParquetRecordBatchReader {
        let f = File::open(path).unwrap_or_else(|e| {
            panic!("open file {}: {e}", path.display());
        });
        ParquetRecordBatchReaderBuilder::try_new(f)
            .unwrap_or_else(|e| panic!("open parquet {}: {e}", path.display()))
            .with_batch_size(1024)
            .build()
            .unwrap_or_else(|e| panic!("build parquet reader {}: {e}", path.display()))
    }
}

impl Iterator for Dataset {
    type Item = DocSample;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        loop {
            if let Some(batch) = &self.batch {
                while self.row < batch.num_rows() {
                    let row = self.row;
                    self.row += 1;
                    if let Some(doc) = doc_at(batch, row) {
                        let text = match self.doc_cap {
                            Some(cap) => truncate_chars(doc, cap).to_string(),
                            None => doc.to_string(),
                        };
                        // 与 nanochat 一致：按 Unicode 字符数累计，本篇仍产出，超出后再停。
                        self.chars_seen += text.chars().count() as u64;
                        let doc_idx = self.doc_idx;
                        self.doc_idx += 1;
                        if self.max_chars.is_some_and(|m| self.chars_seen > m) {
                            self.exhausted = true;
                        }
                        return Some(DocSample {
                            text,
                            doc_idx,
                            shard_idx: self.shard_idx,
                        });
                    }
                }
                self.batch = None;
            }

            if self.reader.is_none() {
                if self.path_idx >= self.paths.len() {
                    return None;
                }
                let (shard_idx, path) = &self.paths[self.path_idx];
                self.shard_idx = *shard_idx;
                self.path_idx += 1;
                self.doc_idx = 0;
                self.reader = Some(Self::open_reader(path));
            }

            match self.reader.as_mut().unwrap().next() {
                Some(Ok(batch)) => {
                    self.row = 0;
                    self.batch = Some(batch);
                }
                Some(Err(e)) => panic!("read record batch failed: {e}"),
                None => {
                    self.reader = None;
                }
            }
        }
    }
}
