use crate::bpe::TokenId;

/// `token_id → bytes`。特殊 token（如 BOS）不在表内，decode 时跳过。
pub struct BpeDecoder {
    id_to_bytes: Vec<Vec<u8>>,
}

impl BpeDecoder {
    /// `id_to_bytes[i]` 为 token `i` 的字节；未使用的下标可为空。
    pub fn new(id_to_bytes: Vec<Vec<u8>>) -> Self {
        Self { id_to_bytes }
    }

    /// 将 token id 序列还原为 UTF-8 文本；跳过表外 id（特殊 token）。
    pub fn decode(&self, ids: &[TokenId]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(piece) = self.id_to_bytes.get(id as usize) else {
                continue;
            };
            if piece.is_empty() {
                continue;
            }
            bytes.extend_from_slice(piece);
        }
        String::from_utf8(bytes).unwrap_or_else(|e| {
            panic!("decode produced invalid UTF-8: {e}");
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::BpeEncoder;
    use crate::paths;
    use ahash::AHashMap;
    use base64::Engine;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    fn load_id_to_bytes(path: &std::path::Path) -> Vec<Vec<u8>> {
        let f = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let mut ranks: AHashMap<Vec<u8>, TokenId> = AHashMap::new();
        let mut max_id = 0u32;
        for line in BufReader::new(f).lines() {
            let line = line.expect("read line");
            let (token_b64, rank) = line.split_once(' ').expect("bad line");
            let id: TokenId = rank.parse().expect("bad rank");
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(token_b64)
                .expect("bad b64");
            max_id = max_id.max(id);
            ranks.insert(bytes, id);
        }
        let mut id_to_bytes = vec![Vec::new(); max_id as usize + 1];
        for (bytes, id) in ranks {
            id_to_bytes[id as usize] = bytes;
        }
        id_to_bytes
    }

    #[test]
    fn decode_encode_roundtrip() {
        let path = paths::vocab_tiktoken_path();
        assert!(
            path.is_file(),
            "missing vocab at {}, run `rust-bpe train` first",
            path.display()
        );

        let encoder = BpeEncoder::from_tiktoken_path(&path);
        let decoder = BpeDecoder::new(load_id_to_bytes(&path));

        let texts = [
            "Hello, world!",
            "你好，世界！",
            "GPT-4 pattern: don't 123\n\tspaces",
            "混合 English 和中文 😊",
            "",
        ];
        for text in texts {
            let ids = encoder.encode(text);
            let out = decoder.decode(&ids);
            assert_eq!(out, text, "roundtrip failed for {text:?}, ids={ids:?}");
        }
    }
}
