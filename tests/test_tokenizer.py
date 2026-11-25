"""Keep the Python/Rust token-format contract covered after removing CORE eval."""
import base64
import json
from pathlib import Path
import subprocess
import pytest
from nanochat.tokenizer import SPECIAL_NAMES, Tokenizer


@pytest.mark.parametrize("text", [
    "",
    "abc ababab 12345",
    "abc abc xyz abc xyz",
    "ab ab\tab\nab",
    "中文和 emoji 😀\nNext line",
    "中文 中文 中文",
    "<|bos|> ordinary text\t",
    "a" * 257,
    "abab" * 64,
    "abcd" * 40,
])
def test_python_rust_encoding_equivalence(tmp_path, text):
    binary = Path(__file__).resolve().parents[1] / "rust-bpe/target/debug/rust-bpe"
    if not binary.exists():
        pytest.skip("build Rust binary with cargo build --locked first")
    ranks = {bytes([i]): i for i in range(256)}
    ranks[b"ab"] = 256
    ranks[b"abc"] = 257
    ranks[b" a"] = 258
    ranks[b" ab"] = 259
    ranks[b" abc"] = 260
    ranks[b"aa"] = 261
    ranks[b"aaa"] = 262
    ranks[b"aaaa"] = 263
    vocab = tmp_path / "vocab.tiktoken"
    vocab.write_text("".join(f"{base64.b64encode(raw).decode()} {rank}\n" for raw, rank in ranks.items()))
    specials = {name: 32759 + i for i, name in enumerate(SPECIAL_NAMES)}
    tokenizer = Tokenizer.load_from_path(vocab, specials)
    result = subprocess.run([str(binary), "encode-text", str(vocab)], input=text, text=True,
                            capture_output=True, check=True)
    assert json.loads(result.stdout) == tokenizer.encode(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text
