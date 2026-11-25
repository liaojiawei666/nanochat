pub mod decode;
pub mod encode;
mod pretok;
pub mod train;

pub use decode::BpeDecoder;
pub use encode::BpeEncoder;
pub use train::BpeTrainer;

/// BPE token id（也用作 mergeable rank）。
pub type TokenId = u32;

/// 相邻两个 token 构成的 pair。
pub type Pair = (TokenId, TokenId);
