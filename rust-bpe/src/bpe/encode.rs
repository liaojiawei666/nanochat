use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use ahash::AHashMap;
use base64::Engine;

use crate::bpe::TokenId;
use crate::bpe::pretok::Pretokenizer;

/// tiktoken 风格：`bytes → rank`（rank 即 token id / 合并优先级）。
type MergeableRanks = AHashMap<Vec<u8>, TokenId>;

pub struct BpeEncoder {
    ranks: MergeableRanks,
    pretokenizer: Pretokenizer,
}

impl BpeEncoder {
    pub fn from_tiktoken_path(path: &Path) -> Self {
        let f = File::open(path).unwrap_or_else(|e| {
            panic!("未找到文件 {}: {e}", path.display());
        });
        let mut ranks: MergeableRanks = AHashMap::new();
        for line in BufReader::new(f).lines() {
            let line = line.expect("read vocab line error");
            let (token_b64, rank) = line.split_once(' ').expect("bad vocab line");
            let id: TokenId = rank.parse().expect("bad rank");
            let bytes: Vec<u8> = base64::engine::general_purpose::STANDARD
                .decode(token_b64)
                .expect("bad token");
            ranks.insert(bytes, id);
        }
        Self {
            ranks,
            pretokenizer: Pretokenizer::new(),
        }
    }

    /// 仅编码正文；相同片段在文档内编码一次，再按原文顺序拼接。
    pub fn encode(&self, doc: &str) -> Vec<TokenId> {
        let mut word_tokens: AHashMap<&str, Vec<TokenId>> = AHashMap::new();
        let mut token_ids = Vec::new();
        for word in self.pretokenizer.split(doc) {
            let ids = word_tokens
                .entry(word)
                .or_insert_with(|| self.encode_word(word));
            token_ids.extend_from_slice(ids);
        }
        token_ids
    }

    fn encode_word(&self, word: &str) -> Vec<TokenId> {
        let bytes = word.as_bytes();
        // 起点 → 终点，按原文位置排列当前片段。
        let mut parts: BTreeMap<usize, usize> = (0..bytes.len()).map(|i| (i, i + 1)).collect();
        // (rank, 起点)：最小 rank 优先，同 rank 时最左优先。
        let mut candidates = BTreeSet::new();
        for &start in parts.keys() {
            if let Some(candidate) = self.pair_candidate(bytes, &parts, start) {
                candidates.insert(candidate);
            }
        }

        while let Some((_, left)) = candidates.pop_first() {
            let right = parts[&left];
            let end = parts[&right];
            let previous = parts.range(..left).next_back().map(|(&start, _)| start);

            // 合并 B、C 前删除 AB、CD；BC 已由 pop_first 取出。
            for start in [previous, Some(right)].into_iter().flatten() {
                if let Some(candidate) = self.pair_candidate(bytes, &parts, start) {
                    candidates.remove(&candidate);
                }
            }

            parts.insert(left, end);
            parts.remove(&right);

            // 合并后添加 A(BC)、(BC)D，其余候选不变。
            for start in [previous, Some(left)].into_iter().flatten() {
                if let Some(candidate) = self.pair_candidate(bytes, &parts, start) {
                    candidates.insert(candidate);
                }
            }
        }

        parts
            .into_iter()
            .map(|(start, end)| self.ranks[&bytes[start..end]])
            .collect()
    }

    fn pair_candidate(
        &self,
        bytes: &[u8],
        parts: &BTreeMap<usize, usize>,
        start: usize,
    ) -> Option<(TokenId, usize)> {
        let right = *parts.get(&start)?;
        let end = *parts.get(&right)?;
        self.ranks
            .get(&bytes[start..end])
            .map(|&rank| (rank, start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder(merges: &[&[u8]]) -> BpeEncoder {
        let mut ranks: MergeableRanks = (0u16..256)
            .map(|id| (vec![id as u8], id as TokenId))
            .collect();
        for (index, bytes) in merges.iter().enumerate() {
            ranks.insert(bytes.to_vec(), 256 + index as TokenId);
        }
        BpeEncoder {
            ranks,
            pretokenizer: Pretokenizer::new(),
        }
    }

    #[test]
    fn merging_middle_pair_updates_both_neighbors() {
        let encoder = encoder(&[b"bc", b"ab", b"cd", b"abc", b"bcd", b"abcd"]);
        assert_eq!(encoder.encode_word("abcd"), [261]);
    }

    #[test]
    fn equal_rank_merges_choose_leftmost_pair() {
        let encoder = encoder(&[b"aa", b"aaa"]);
        assert_eq!(encoder.encode_word("aaaaa"), [256, 257]);
    }
}
