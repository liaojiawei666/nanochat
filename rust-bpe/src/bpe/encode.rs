use std::{
    collections::BinaryHeap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use ahash::AHashMap;
use base64::Engine;
use fancy_regex::Regex;

use crate::bpe::TokenId;
use crate::special::{GPT4_PATTERN, SpecialToken};

/// tiktoken 风格：`bytes → rank`（rank 即 token id / 合并优先级）。
type MergeableRanks = AHashMap<Vec<u8>, TokenId>;

pub struct BpeEncoder {
    ranks: MergeableRanks,
    pattern: Regex,
}

/// 链表节点下标（合并后仍指向该段第一个字节的下标）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct NodeId(usize);

/// 堆中的待合并相邻段；`rank` 越小越优先。
#[derive(Debug, PartialEq, Eq)]
struct MergeCandidate {
    left: NodeId,
    right: NodeId,
    rank: TokenId,
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap 是大根堆：反过来比 => rank 小的优先；同 rank 时 left 小的优先
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
    }
}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct TokenNode {
    alive: bool,
    token_id: TokenId,
    prev: Option<NodeId>,
    next: Option<NodeId>,
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
            pattern: Regex::new(GPT4_PATTERN).unwrap(),
        }
    }

    pub fn encode(&self, doc: &str) -> Vec<TokenId> {
        let words: Vec<&str> = self
            .pattern
            .find_iter(doc)
            .map(|m| m.expect("regex match failed").as_str())
            .collect();
        let mut token_ids: Vec<TokenId> = Vec::new();
        token_ids.push(SpecialToken::Bos.id());
        for word in words {
            token_ids.extend(self.encode_word(word));
        }
        token_ids
    }

    fn encode_word(&self, word: &str) -> Vec<TokenId> {
        let bytes: Vec<u8> = word.bytes().collect();
        let total_bytes = bytes.len();
        if total_bytes == 0 {
            return Vec::new();
        }

        let mut pair_heap: BinaryHeap<MergeCandidate> = BinaryHeap::new();
        let mut nodes: Vec<TokenNode> = Vec::with_capacity(total_bytes);

        for i in 0..total_bytes {
            let mut node = TokenNode {
                token_id: bytes[i] as TokenId,
                prev: None,
                next: None,
                alive: true,
            };
            if i > 0 {
                node.prev = Some(NodeId(i - 1));
            }
            if i + 1 < total_bytes {
                node.next = Some(NodeId(i + 1));
                if let Some(&rank) = self.ranks.get(&bytes[i..i + 2]) {
                    pair_heap.push(MergeCandidate {
                        left: NodeId(i),
                        right: NodeId(i + 1),
                        rank,
                    });
                }
            }
            nodes.push(node);
        }

        while let Some(MergeCandidate {
            left,
            right,
            rank,
        }) = pair_heap.pop()
        {
            if !(nodes[left.0].alive
                && nodes[right.0].alive
                && nodes[left.0].next == Some(right))
            {
                continue;
            }

            let byte_end = nodes[right.0].next.map_or(total_bytes, |n| n.0);
            match self.ranks.get(&bytes[left.0..byte_end]) {
                Some(&r) if r == rank => {}
                _ => continue,
            }

            nodes[right.0].alive = false;
            nodes[left.0].token_id = rank;
            let new_right_node_id = nodes[right.0].next;
            nodes[left.0].next = new_right_node_id;

            if let Some(new_right) = new_right_node_id {
                nodes[new_right.0].prev = Some(left);
                let pair_byte_end = nodes[new_right.0].next.map_or(total_bytes, |n| n.0);
                if let Some(&tok) = self.ranks.get(&bytes[left.0..pair_byte_end]) {
                    pair_heap.push(MergeCandidate {
                        left,
                        right: new_right,
                        rank: tok,
                    });
                }
            }

            if let Some(prev_left) = nodes[left.0].prev {
                let pair_byte_end = nodes[left.0].next.map_or(total_bytes, |n| n.0);
                if let Some(&tok) = self.ranks.get(&bytes[prev_left.0..pair_byte_end]) {
                    pair_heap.push(MergeCandidate {
                        left: prev_left,
                        right: left,
                        rank: tok,
                    });
                }
            }
        }

        let mut token_ids = Vec::new();
        let mut pos = NodeId(0);
        loop {
            token_ids.push(nodes[pos.0].token_id);
            match nodes[pos.0].next {
                Some(n) => pos = n,
                None => break,
            }
        }
        token_ids
    }
}

