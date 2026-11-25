use fancy_regex::Regex;

const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// 训练和编码共用的预分词器，返回借用原文的文本片段。
pub struct Pretokenizer {
    pattern: Regex,
}

impl Pretokenizer {
    pub fn new() -> Self {
        Self {
            pattern: Regex::new(GPT4_PATTERN).expect("invalid GPT4 regex"),
        }
    }

    pub fn split<'a>(&'a self, text: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.pattern
            .find_iter(text)
            .map(|m| m.expect("regex match failed").as_str())
    }
}
