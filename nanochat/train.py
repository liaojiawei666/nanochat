from nanochat.dataloader import ClimbmixDataLoader, ClimbmixDataset
from nanochat.gpt import GPTConfig, GPT
import torch
import torch.nn.functional as F

n_layer = 24
# 累计训练 token ≈ ratio × 参数量（对齐 nanochat 量级，可再调）
TOKEN_PARAM_RATIO = 8


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPTConfig(n_layer)
    gpt = GPT(
        config.sequence_len,
        config.vocab_size,
        config.n_embd,
        config.head_dim,
        config.n_kv_head,
        config.n_layer,
    ).to(device)

    dataset = ClimbmixDataset("train")
    loader = ClimbmixDataLoader(
        dataset, B=config.batch, T=config.sequence_len, device=device
    )
    n_tokens = 0
    for x_idx, y_idx in loader:
        n_tokens = n_tokens + x_idx.numel()
        if n_tokens >= config.n_tokens:
            break
        logits = gpt(x_idx)  # (B,T,vocab_size)
        logits = config.softcap * torch.tanh(logits / config.softcap)
        loss=F.cross_entropy(logits.view(-1,logits.size(-1)),y_idx.view(-1))
        loss.backward()
        


if __name__ == "__main__":
    train()
