import io
from pathlib import Path
import tarfile
import numpy as np
import pytest
import torch
import zstandard as zstd
from script.data import pack_dataset, extract_dataset
from nanochat.dataloader import ClimbmixDataset, ClimbmixDataLoader, list_token_shards
from nanochat.tokenizer import SPECIAL_TOKENS


def test_pack_extract_roundtrip_and_split(tiny_config, tmp_path):
    cfg = tiny_config.storage
    source = Path(cfg.data_dir)
    pack_dataset(cfg)
    cfg.data_dir = str(tmp_path / "extracted")
    root = extract_dataset(cfg)
    assert {p.name for p in root.iterdir()} == {p.name for p in source.iterdir()}
    for path in root.iterdir():
        assert path.read_bytes() == (source / path.name).read_bytes()
    train_doc = next(iter(ClimbmixDataset("train", root)))
    val_doc = next(iter(ClimbmixDataset("val", root)))
    assert train_doc[1] == ord("a") and val_doc[1] == ord("q")
    (root / "stale" / "nested").mkdir(parents=True)
    (root / "stale" / "nested" / "old.bin").write_bytes(b"old data")
    (root / "shard_99999.bin").write_bytes(b"stale shard")
    (root / "vocab.tiktoken").write_bytes(b"stale vocabulary")
    extract_dataset(cfg)
    assert {p.name for p in root.iterdir()} == {p.name for p in source.iterdir()}
    for path in root.iterdir():
        assert path.read_bytes() == (source / path.name).read_bytes()


def test_document_packing_and_shift(tiny_config):
    root = Path(tiny_config.storage.data_dir)
    bos = SPECIAL_TOKENS["<|bos|>"]
    np.asarray([bos, 1, 2, bos, 3, 4, 5, 6, 7, bos, 8, 9, 10], dtype="<u2").tofile(root / "shard_00000.bin")
    loader = ClimbmixDataLoader(ClimbmixDataset("train", root), B=1, T=6, buffer_size=3, device="cpu")
    x, y = next(loader)
    # Best fit takes the six-token document, then truncates the shortest one.
    assert x.tolist() == [[bos, 3, 4, 5, 6, 7]]
    assert y.tolist() == [[3, 4, 5, 6, 7, bos]]
    assert x.device.type == "cpu" and x.dtype == torch.int64
    for _ in range(8):
        assert next(loader)[0].shape == (1, 6)


def test_shards_are_sorted_and_partial_files_are_ignored(tiny_config):
    root = Path(tiny_config.storage.data_dir)
    (root / "shard_99999.bin.part").write_bytes(b"partial")
    (root / "notes.txt").write_text("not a shard")
    (root / "shard_00000.bin").rename(root / "shard_00010.bin")
    shards = list_token_shards(root)
    assert [path.name for path in shards] == ["shard_00001.bin", "shard_00002.bin", "shard_00010.bin"]
    assert ClimbmixDataset("train", root).paths == shards[:-1]
    assert ClimbmixDataset("val", root).paths == shards[-1:]


def test_dataset_requires_training_and_validation_shards(tiny_config):
    root = Path(tiny_config.storage.data_dir)
    for path in sorted(root.glob("*.bin"))[1:]:
        path.unlink()
    with pytest.raises(ValueError, match="at least two"):
        list_token_shards(root)


@pytest.mark.parametrize("name", ["../outside.bin", "/outside.bin", "link.bin"])
def test_archive_cannot_write_outside_destination(tiny_config, tmp_path, name):
    cfg = tiny_config.storage
    cfg.data_dir = str(tmp_path / "extracted")
    Path(cfg.data_dir).mkdir()
    (Path(cfg.data_dir) / "old.bin").write_bytes(b"old data")
    with open(cfg.archive_path, "wb") as raw, zstd.ZstdCompressor().stream_writer(raw) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as tar:
            member = tarfile.TarInfo(name)
            if name == "link.bin":
                member.type, member.linkname = tarfile.SYMTYPE, "../outside.bin"
            else:
                member.size = 1
            tar.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        extract_dataset(cfg)
    assert not Path(cfg.data_dir).exists() and not (tmp_path / "outside.bin").exists()


def test_missing_archive_does_not_clear_destination(tiny_config, tmp_path):
    cfg = tiny_config.storage
    root = Path(cfg.data_dir)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    cfg.archive_path = str(tmp_path / "missing.tar.zst")
    with pytest.raises(FileNotFoundError):
        extract_dataset(cfg)
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


def test_archive_inside_destination_is_rejected_before_deletion(tiny_config):
    cfg = tiny_config.storage
    root = Path(cfg.data_dir)
    cfg.archive_path = str(root / "data.tar.zst")
    pack_dataset(cfg)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(ValueError, match="outside data_dir"):
        extract_dataset(cfg)
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


def test_symlink_destination_is_rejected_before_deletion(tiny_config, tmp_path):
    cfg = tiny_config.storage
    source = Path(cfg.data_dir)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    destination = tmp_path / "linked tokens"
    destination.symlink_to(source, target_is_directory=True)
    cfg.data_dir = str(destination)
    with pytest.raises(ValueError, match="symlink"):
        extract_dataset(cfg)
    assert destination.is_symlink()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
