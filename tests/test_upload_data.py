"""使用临时目录和本地 Bash 验证代码预览、压缩包上传与校验，不连接服务器。"""
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from types import SimpleNamespace
from typing import cast

import paramiko
import pytest

from script import upload_data as data_upload
from script.upload_data import upload_data
from script.remote import run



class LocalSFTP:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def stat(self, path):
        return Path(path).stat()

    def mkdir(self, path):
        Path(path).mkdir()

    def put(self, source, destination, callback=None):
        shutil.copyfile(source, destination)
        if callback:
            size = Path(source).stat().st_size
            callback(size, size)

    def chmod(self, path, mode):
        Path(path).chmod(mode)

    def posix_rename(self, source, destination):
        os.replace(source, destination)


class CommandOutput(io.BytesIO):
    def __init__(self, text: str, status: int):
        super().__init__(text.encode())
        self.channel = SimpleNamespace(recv_exit_status=lambda: status, close=lambda: None)


class LocalSSH:
    def open_sftp(self):
        return LocalSFTP()

    def exec_command(self, command):
        result = subprocess.run(["bash", "-c", command], capture_output=True, check=False)
        output = CommandOutput(result.stdout.decode(), result.returncode)
        return io.BytesIO(), output, io.BytesIO(result.stderr)


def local_ssh() -> paramiko.SSHClient:
    # 用本地执行器实现测试所需的 SSH/SFTP 接口，替代真实网络连接。
    return cast(paramiko.SSHClient, LocalSSH())


def test_remote_run_preserves_output_on_failure():
    with pytest.raises(RuntimeError, match="status 7") as error:
        run(local_ssh(), "echo standard; echo failure >&2; exit 7")
    assert "standard" in str(error.value) and "failure" in str(error.value)


def test_data_upload_is_verified_and_skips_identical_archive(tmp_path, capsys):
    source = tmp_path / "archive $(echo harmless).tar.zst"
    source.write_bytes(b"new dataset")
    target = tmp_path / "shared space/data.tar.zst"
    upload_data(local_ssh(), source, PurePosixPath(target))
    assert target.read_bytes() == source.read_bytes()
    assert not target.with_name(target.name + ".part").exists()
    upload_data(local_ssh(), source, PurePosixPath(target))
    assert "already uploaded" in capsys.readouterr().out


def test_failed_upload_does_not_replace_existing_archive(tmp_path, monkeypatch):
    source = tmp_path / "data.tar.zst"
    source.write_bytes(b"new dataset")
    target = tmp_path / "shared/data.tar.zst"
    target.parent.mkdir()
    target.write_bytes(b"previous archive")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(data_upload, "remote_digest", lambda *args: "different from " + digest)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        upload_data(local_ssh(), source, PurePosixPath(target))
    assert target.read_bytes() == b"previous archive"


def test_remote_digest_tolerates_login_messages(tmp_path):
    source = tmp_path / "archive with spaces"
    source.write_bytes(b"dataset")

    class NoisySSH(LocalSSH):
        def exec_command(self, command):
            return super().exec_command("echo login-message\n" + command)

    digest = data_upload.remote_digest(cast(paramiko.SSHClient, NoisySSH()), PurePosixPath(source))
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()


def test_interrupted_upload_keeps_previous_archive(tmp_path):
    source = tmp_path / "data.tar.zst"
    source.write_bytes(b"new dataset")
    target = tmp_path / "shared/data.tar.zst"
    target.parent.mkdir()
    target.write_bytes(b"previous archive")

    class InterruptedSFTP(LocalSFTP):
        def put(self, source, destination, callback=None):
            Path(destination).write_bytes(b"partial")
            raise OSError("connection interrupted")

    class InterruptedSSH(LocalSSH):
        def open_sftp(self):
            return InterruptedSFTP()

    with pytest.raises(OSError, match="interrupted"):
        upload_data(cast(paramiko.SSHClient, InterruptedSSH()), source, PurePosixPath(target))
    assert target.read_bytes() == b"previous archive"
    upload_data(local_ssh(), source, PurePosixPath(target))
    assert target.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("platform", ["AutoDL", "DAMODEL"])
@pytest.mark.parametrize("command", ["code", "data"])
def test_every_platform_command_can_be_previewed_without_credentials(platform, command, tmp_path, monkeypatch, capsys):
    from script import platform_cli

    root = Path(__file__).resolve().parents[1]
    for name in ("config.json",):
        shutil.copyfile(root / name, tmp_path / name)

    def unexpected_connection(**kwargs):
        raise AssertionError("dry run must not connect")

    monkeypatch.setattr(platform_cli, "connect", unexpected_connection)
    platform_cli.main(platform, tmp_path, [command, "--dry-run"])
    preview = capsys.readouterr().out
    assert preview


def test_unmounted_shared_storage_is_rejected(tmp_path):
    from script.remote import require_shared_storage

    with pytest.raises(RuntimeError, match="Shared storage is not mounted"):
        require_shared_storage(local_ssh(), PurePosixPath(tmp_path))
