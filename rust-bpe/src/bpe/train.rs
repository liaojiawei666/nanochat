use ahash::{AHashMap, AHashSet};
use base64::Engine;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::bpe::pretok::Pretokenizer;
use crate::bpe::{Pair, TokenId};

/// 在调用方提供的文档迭代器上统计 pretok 片段全局频次。
fn count_pretok_strs(documents: impl Iterator<Item = String> + Send) -> AHashMap<String, i64> {
    let pretokenizer = Pretokenizer::new();
    documents
        .par_bridge()
        .map(|text| {
            let mut counts = AHashMap::new();
            for piece in pretokenizer.split(&text) {
                *counts.entry(piece.to_string()).or_insert(0) += 1;
            }
            counts
        })
        .reduce(AHashMap::new, |mut acc, local| {
            for (word, n) in local {
                *acc.entry(word).or_insert(0) += n;
            }
            acc
        })
}

/// pretok 片段：字节级 token 序列 + 在语料中的出现次数。
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
struct Word {
    ids: Vec<TokenId>,
    count: i64,
}

fn add_delta(delta: &mut AHashMap<Pair, i64>, pair: Pair, amount: i64) {
    *delta.entry(pair).or_insert(0) += amount;
}

impl Word {
    fn pairs(&self) -> impl Iterator<Item = Pair> + '_ {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    /// 从左到右不重叠地合并 `pair`，返回次数变化和合并后仍存在的 pair。
    fn merge(&mut self, pair: Pair, token_id: TokenId) -> (AHashMap<Pair, i64>, AHashSet<Pair>) {
        let mut delta: AHashMap<Pair, i64> = AHashMap::new();
        let mut new_ids: Vec<TokenId> = Vec::new();
        let mut cur = 0;
        let n = self.ids.len();
        while cur < n {
            if cur + 1 < n && self.ids[cur] == pair.0 && self.ids[cur + 1] == pair.1 {
                if let Some(prev) = new_ids.last().copied() {
                    add_delta(&mut delta, (prev, self.ids[cur]), -self.count);
                    add_delta(&mut delta, (prev, token_id), self.count);
                }
                if cur + 2 < n {
                    add_delta(
                        &mut delta,
                        (self.ids[cur + 1], self.ids[cur + 2]),
                        -self.count,
                    );
                    add_delta(&mut delta, (token_id, self.ids[cur + 2]), self.count);
                }
                new_ids.push(token_id);
                cur += 2;
            } else {
                new_ids.push(self.ids[cur]);
                cur += 1;
            }
        }
        self.ids = new_ids;
        let pairs_in_word = self.pairs().collect();
        (delta, pairs_in_word)
    }
}

#[derive(Default)]
struct PairStat {
    count: i64,
    pos: AHashSet<usize>,
}

fn build_words(str_count: &AHashMap<String, i64>) -> Vec<Word> {
    str_count
        .par_iter()
        .map(|(s, &count)| Word {
            ids: s.as_bytes().iter().map(|&b| b as TokenId).collect(),
            count,
        })
        .collect()
}

fn build_pair_stats(words: &[Word]) -> AHashMap<Pair, PairStat> {
    words
        .par_iter()
        .enumerate()
        .fold(
            || AHashMap::<Pair, PairStat>::new(),
            |mut local, (pos, word)| {
                for pair in word.pairs() {
                    let stat = local.entry(pair).or_default();
                    stat.pos.insert(pos);
                    stat.count += word.count;
                }
                local
            },
        )
        .reduce(AHashMap::new, |mut acc, local| {
            for (pair, stat2) in local {
                let stat1 = acc.entry(pair).or_default();
                stat1.pos.extend(stat2.pos);
                stat1.count += stat2.count;
            }
            acc
        })
}

/// 大根堆：先比 count；同频时 `Reverse(pair)` 让更小的 pair 先出。
type PairHeap = BinaryHeap<(i64, Reverse<Pair>)>;

fn build_pair_heap(pair_stats: &AHashMap<Pair, PairStat>) -> PairHeap {
    let mut heap = BinaryHeap::new();
    for (pair, stat) in pair_stats {
        heap.push((stat.count, Reverse(*pair)));
    }
    heap
}

pub struct BpeTrainer {
    words: Vec<Word>,
    pair_stats: AHashMap<Pair, PairStat>,
    heap: PairHeap,
    id_to_bytes: Vec<Vec<u8>>,
}

impl BpeTrainer {
    pub fn from_documents(documents: impl Iterator<Item = String> + Send) -> Self {
        let str_count = count_pretok_strs(documents);
        let words = build_words(&str_count);
        let pair_stats = build_pair_stats(&words);
        let heap = build_pair_heap(&pair_stats);
        let id_to_bytes = (0u8..=255).map(|b| vec![b]).collect();
        Self {
            words,
            pair_stats,
            heap,
            id_to_bytes,
        }
    }

    pub fn train(&mut self, vocab_size: usize) {
        while self.id_to_bytes.len() < vocab_size {
            let (pair, positions) = self.pop_fresh_pair();
            let token_id = self.push_token(pair);
            let changed = self.apply_merge(pair, token_id, &positions);
            self.refresh_heap(changed);
            self.pair_stats.remove(&pair);
            self.maybe_rebuild_heap();
        }
    }

    /// 弹出次数仍与统计表一致的 pair。堆里的旧记录直接丢掉。
    fn pop_fresh_pair(&mut self) -> (Pair, AHashSet<usize>) {
        loop {
            let (count, Reverse(pair)) = self.heap.pop().expect("pair heap empty");
            if let Some(stat) = self.pair_stats.get(&pair) {
                if stat.count == count {
                    return (pair, stat.pos.clone());
                }
            }
        }
    }

    fn push_token(&mut self, pair: Pair) -> TokenId {
        let mut bytes = self.id_to_bytes[pair.0 as usize].clone();
        bytes.extend_from_slice(&self.id_to_bytes[pair.1 as usize]);
        let token_id = self.id_to_bytes.len() as TokenId;
        self.id_to_bytes.push(bytes);
        token_id
    }

    fn apply_merge(
        &mut self,
        pair: Pair,
        token_id: TokenId,
        positions: &AHashSet<usize>,
    ) -> AHashSet<Pair> {
        let mut changed = AHashSet::new();
        for &word_pos in positions {
            let (delta, pairs_in_word) = self.words[word_pos].merge(pair, token_id);
            for (p, c) in delta {
                let stat = self.pair_stats.entry(p).or_default();
                stat.count += c;
                if pairs_in_word.contains(&p) {
                    stat.pos.insert(word_pos);
                } else {
                    stat.pos.remove(&word_pos);
                }
                changed.insert(p);
            }
        }
        changed
    }

    fn refresh_heap(&mut self, changed: AHashSet<Pair>) {
        for pair in changed {
            let count = self.pair_stats.get(&pair).unwrap().count;
            if count == 0 {
                self.pair_stats.remove(&pair);
            } else {
                self.heap.push((count, Reverse(pair)));
            }
        }
    }

    fn maybe_rebuild_heap(&mut self) {
        if self.heap.len() > 10000 && self.heap.len() > 3 * self.pair_stats.len() {
            self.heap = build_pair_heap(&self.pair_stats);
        }
    }

    pub fn write_vocab(&self, path: &Path) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .unwrap_or_else(|e| panic!("create {}: {e}", parent.display()));
        }
        let file = File::create(path).unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
        let mut out = BufWriter::new(file);
        let mut n = 0usize;
        for (id, bytes) in self.id_to_bytes.iter().enumerate() {
            let token = base64::engine::general_purpose::STANDARD.encode(bytes);
            writeln!(out, "{token} {id}").expect("write vocab");
            n += 1;
        }
        out.flush().expect("flush vocab");
        eprintln!("wrote {n} tokens to {}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_documents_counts_pieces_and_preserves_boundaries() {
        let documents = ["ab cd", "ab cd"].into_iter().map(String::from);
        let mut trainer = BpeTrainer::from_documents(documents);
        let counts: AHashMap<String, i64> = trainer
            .words
            .iter()
            .map(|word| {
                let bytes: Vec<u8> = word.ids.iter().map(|&id| id as u8).collect();
                (String::from_utf8(bytes).unwrap(), word.count)
            })
            .collect();
        assert_eq!(
            counts,
            AHashMap::from_iter([("ab".into(), 2), (" cd".into(), 2)])
        );

        trainer.train(259);
        assert_eq!(
            trainer.id_to_bytes[256..],
            [b" c".to_vec(), b"ab".to_vec(), b" cd".to_vec()]
        );
    }
}
