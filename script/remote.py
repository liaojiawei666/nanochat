"""通用 SSH/SFTP 操作，不读取平台配置。"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import paramiko
from shlex import quote


@contextmanager
def connect(host: str, port: int, user: str, passwd: str) -> Iterator[paramiko.SSHClient]:
    with paramiko.SSHClient() as ssh:
        # 沿用本机 ssh 的 known_hosts 校验，不自动信任未知服务器。
        ssh.load_system_host_keys()
        ssh.connect(
            hostname=host,
            port=port,
            username=user,
            password=passwd,
            allow_agent=False,
            look_for_keys=False,
            timeout=15,
            banner_timeout=15,
            auth_timeout=15,
        )
        yield ssh


def mkdirs(sftp: paramiko.SFTPClient, path: PurePosixPath) -> None:
    try:
        sftp.stat(str(path))
    except FileNotFoundError:
        mkdirs(sftp, path.parent)
        sftp.mkdir(str(path))


def upload_file(sftp: paramiko.SFTPClient, local: Path, destination: PurePosixPath, callback=None) -> None:
    mkdirs(sftp, destination.parent)
    if callback is None:
        sftp.put(str(local), str(destination))
    else:
        sftp.put(str(local), str(destination), callback=callback)
    sftp.chmod(str(destination), local.stat().st_mode & 0o777)


def run(ssh: paramiko.SSHClient, command: str) -> str:
    """执行挂载检查或文件校验，返回输出，失败时附上错误。"""
    script = "set -euo pipefail\n" + command
    stdin, stdout, stderr = ssh.exec_command(f"bash -lc {quote(script)} 2>&1")
    stdin.close()
    try:
        result = stdout.read().decode("utf-8", errors="replace")
        result += stderr.read().decode("utf-8", errors="replace")
        status = stdout.channel.recv_exit_status()
        if status:
            raise RuntimeError(f"remote command exited with status {status}\n{result}")
        return result
    finally:
        stdout.channel.close()


def require_shared_storage(ssh: paramiko.SSHClient, shared_dir: PurePosixPath) -> None:
    run(ssh, f"mountpoint -q {quote(str(shared_dir))} || "
        "{ echo 'Shared storage is not mounted; mount it in the platform console first.' >&2; exit 1; }")
