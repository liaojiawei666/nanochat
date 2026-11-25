use arrow::array::{Array, AsArray, StringArrayType};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use std::fs::File;
use std::path::{Path, PathBuf};

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

/// 按传入的 shard 顺序读取完整文档；跳过 NULL，保留空字符串。
pub struct Dataset {
    shard_list: std::vec::IntoIter<PathBuf>,
    reader: Option<ParquetRecordBatchReader>,
    batch: Option<RecordBatch>,
    row: usize,
}

impl Dataset {
    pub fn new(shard_list: Vec<PathBuf>) -> Self {
        Self {
            shard_list: shard_list.into_iter(),
            reader: None,
            batch: None,
            row: 0,
        }
    }

    fn open_reader(path: &Path) -> ParquetRecordBatchReader {
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
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(batch) = &self.batch {
                while self.row < batch.num_rows() {
                    let row = self.row;
                    self.row += 1;
                    if let Some(doc) = doc_at(batch, row) {
                        return Some(doc.to_string());
                    }
                }
                self.batch = None;
            }

            if self.reader.is_none() {
                let path = self.shard_list.next()?;
                self.reader = Some(Self::open_reader(&path));
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
