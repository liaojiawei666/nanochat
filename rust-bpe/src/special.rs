/// 文档分隔和微调对话用的特殊 token。声明顺序就是 id 顺序。
#[derive(Clone, Copy)]
pub enum SpecialToken {
    /// 每篇文档开头的 BOS，用来分隔文档。
    Bos,
    UserStart,
    UserEnd,
    AssistantStart,
    AssistantEnd,
    PythonStart,
    PythonEnd,
    OutputStart,
    OutputEnd,
}

impl SpecialToken {
    pub const ALL: [SpecialToken; 9] = [
        Self::Bos,
        Self::UserStart,
        Self::UserEnd,
        Self::AssistantStart,
        Self::AssistantEnd,
        Self::PythonStart,
        Self::PythonEnd,
        Self::OutputStart,
        Self::OutputEnd,
    ];

    pub const fn id(self) -> u32 {
        VOCAB_NO_SPECIAL_SIZE as u32 + self as u32
    }
}

pub const VOCAB_NO_SPECIAL_SIZE: usize = (1usize << 15) - SpecialToken::ALL.len();
