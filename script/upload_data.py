"""上传编码数据压缩包，校验完整性后发布。"""
import hashlib
from pathlib import Path, PurePosixPath
from shlex import quote
import time

import paramiko

from script.remote import run, upload_file


def remote_digest(ssh: paramiko.SSHClient, path: PurePosixPath) -> str:
    # 登录脚本可能先打印提示；从最后一行读取摘要，文件名不会进入输出。
    return run(ssh, f"sha256sum < {quote(str(path))}").splitlines()[-1].split()[0]


def upload_data(ssh: paramiko.SSHClient, local_archive: Path, remote_archive: PurePosixPath) -> None:
    if not local_archive.is_file():
        raise FileNotFoundError(f"{local_archive}; create it with: python -m script.data pack")
    print(f"hashing {local_archive}", flush=True)
    with local_archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    temporary = remote_archive.with_name(remote_archive.name + ".part")
    last_report = 0.0

    def progress(done: int, total: int) -> None:
        nonlocal last_report
        now = time.monotonic()
        if now - last_report >= 5 or done == total:
            print(f"uploaded {done / 2**20:.1f}/{total / 2**20:.1f} MiB", flush=True)
            last_report = now

    with ssh.open_sftp() as sftp:
        try:
            existing_size = sftp.stat(str(remote_archive)).st_size
        except FileNotFoundError:
            existing_size = None
        if existing_size == local_archive.stat().st_size and remote_digest(ssh, remote_archive) == digest:
            print(f"already uploaded: {remote_archive}")
            return
        upload_file(sftp, local_archive, temporary, callback=progress)
        if remote_digest(ssh, temporary) != digest:
            raise RuntimeError("uploaded archive checksum mismatch; final archive was not replaced")
        sftp.posix_rename(str(temporary), str(remote_archive))
    print(f"verified and uploaded: {remote_archive}")
