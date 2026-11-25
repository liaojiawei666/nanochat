"""本地打包编码数据，或在服务器解压到训练目录。"""
from __future__ import annotations
import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path
import zstandard as zstd
from nanochat.config import Config, StorageConfig
from nanochat.dataloader import list_token_shards


def pack_dataset(cfg: StorageConfig) -> Path:
    root, archive = Path(cfg.data_dir), Path(cfg.archive_path)
    shards = list_token_shards(root)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".part")
    try:
        with temporary.open("wb") as raw, zstd.ZstdCompressor(level=3).stream_writer(raw) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as tar:
                for path in [root / "vocab.tiktoken", *shards]:
                    tar.add(path, arcname=path.name, recursive=False)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def extract_dataset(cfg: StorageConfig) -> Path:
    root, archive = Path(cfg.data_dir), Path(cfg.archive_path)
    destination = root.resolve()
    if root.is_symlink() or destination in Path(__file__).resolve().parents or destination == Path.cwd().resolve():
        raise ValueError(f"data_dir is a protected directory or symlink: {root}")
    if archive.resolve().is_relative_to(destination):
        raise ValueError("archive_path must be outside data_dir before clearing the destination")
    # 先确认压缩包可读，再清除整个目标目录；失败时不会保留旧数据。
    with archive.open("rb") as raw:
        if root.exists():
            shutil.rmtree(root)
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=root.parent) as temporary:
            stage = Path(temporary) / "data"
            stage.mkdir()
            with zstd.ZstdDecompressor().stream_reader(raw) as stream:
                with tarfile.open(fileobj=stream, mode="r|") as tar:
                    for member in tar:
                        name = member.name
                        if (not member.isfile() or Path(name).name != name or "\\" in name
                                or (stage / name).exists() or not (
                                    name == "vocab.tiktoken" or (
                                        name.startswith("shard_") and name.endswith(".bin")))):
                            raise ValueError(f"unexpected dataset archive member: {name}")
                        source = tar.extractfile(member)
                        if source is None:
                            raise ValueError(f"unreadable archive member: {name}")
                        with source, (stage / name).open("wb") as target:
                            shutil.copyfileobj(source, target)
            list_token_shards(stage)
            stage.rename(root)
    return root


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m script.data")
    parser.add_argument("action", choices=["pack", "extract"], help="extract 会先清空目标数据目录")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path, help="覆盖 config.json 中的编码数据目录")
    parser.add_argument("--archive-path", type=Path, help="覆盖压缩包路径")
    args = parser.parse_args(argv)
    cfg = Config.load(args.config).storage
    if args.data_dir is not None:
        cfg.data_dir = str(args.data_dir)
    if args.archive_path is not None:
        cfg.archive_path = str(args.archive_path)
    result = pack_dataset(cfg) if args.action == "pack" else extract_dataset(cfg)
    print(result)


if __name__ == "__main__":
    main()
