"""筛选并上传项目代码；连接和目标目录由平台入口传入。"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

import paramiko

from script.remote import upload_file


CODE_PATTERNS = (
    "nanochat/**/*.py",
    "script/*.py",
    "config.json",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "README.md",
)


def code_files(root: Path) -> list[Path]:
    """仅选择源码和项目配置，排除 env.json、数据、本地环境和软链接。"""
    return sorted({
        path.relative_to(root)
        for pattern in CODE_PATTERNS
        for path in root.glob(pattern)
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
    })


def upload_code(ssh: paramiko.SSHClient, local_root: Path, project_dir: PurePosixPath,
                config_path: Path | None = None) -> None:
    files = code_files(local_root)
    with ssh.open_sftp() as sftp:
        for relative in files:
            source = config_path if relative == Path("config.json") and config_path is not None else local_root / relative
            upload_file(sftp, source, project_dir / relative.as_posix())
            print(f"uploaded {relative.as_posix()}", flush=True)
    print(f"uploaded {len(files)} files to {project_dir}")
