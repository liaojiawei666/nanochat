"""平台入口只负责上传代码和数据压缩包。"""
import argparse
from functools import partial
import json
from pathlib import Path, PurePosixPath

from script.upload_data import upload_data
from script.remote import connect, require_shared_storage
from script.upload_code import code_files, upload_code


def remote_path(value: str | PurePosixPath, name: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
        raise ValueError(f"{name} must be an absolute project directory or file path")
    return path


def main(platform: str, root: Path, argv=None) -> None:
    parser = argparse.ArgumentParser(prog=f"python -m script.{platform.lower()}")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("code", "data"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=root / "config.json")
        command.add_argument("--env", type=Path, default=root / "env.json")
        command.add_argument("--dry-run", action="store_true", help="只预览，不连接服务器")
        if name == "data":
            command.add_argument("--archive", type=Path, help="本地数据压缩包，默认 storage.archive_path")
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text())
    paths = settings["platforms"][platform]
    shared_dir = None

    if args.command == "code":
        project_dir = remote_path(paths["project_dir"], f"platforms.{platform}.project_dir")
        files = code_files(root)
        preview = "\n".join([*(p.as_posix() for p in files), f"{len(files)} files -> {project_dir}"])
        operation = partial(upload_code, local_root=root, project_dir=project_dir, config_path=args.config)
    else:
        local_archive = root / (args.archive or Path(settings["storage"]["archive_path"]))
        if not args.dry_run and not local_archive.is_file():
            raise FileNotFoundError(f"{local_archive}; create it with: python -m script.data pack")
        archive = remote_path(paths["archive_path"], "archive_path")
        shared_dir = remote_path(paths["shared_dir"], "shared_dir")
        preview = f"{local_archive} -> {archive} (upload and verify SHA256)"
        operation = partial(upload_data, local_archive=local_archive, remote_archive=archive)
    if args.dry_run:
        print(preview)
        return
    login = json.loads(args.env.read_text())[platform]["ssh"]
    with connect(**login) as ssh:
        if shared_dir is not None:
            require_shared_storage(ssh, shared_dir)
        operation(ssh)
