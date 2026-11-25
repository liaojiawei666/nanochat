import json
from pathlib import PurePosixPath
from unittest.mock import MagicMock, call

import pytest

from script import autodl, platform_cli, remote
from script.upload_code import code_files


def test_upload_selection_excludes_local_data_and_credentials(tmp_path):
    included = {"nanochat/train.py", "script/autodl.py", "uv.lock", "config.json"}
    excluded = {"configs/autodl.toml", "env.json", "data/train.bin", ".venv/lib/site.py", "local/vocab.tiktoken",
                "nanochat/__pycache__/cached.py", "script/credentials.json"}
    for name in included | excluded:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test")
    (tmp_path / "nanochat/linked.py").symlink_to(tmp_path / "env.json")
    assert {p.as_posix() for p in code_files(tmp_path)} == included


def test_dry_run_uses_platform_path_without_connecting(tmp_path, monkeypatch, capsys):
    (tmp_path / "pyproject.toml").write_text("[project]")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"platforms": {"AutoDL": {"project_dir": "/srv/shared/nanochat"}}}))
    monkeypatch.setattr(autodl, "ROOT", tmp_path)

    def unexpected_connection(**kwargs):
        raise AssertionError("dry run must not connect")

    monkeypatch.setattr(platform_cli, "connect", unexpected_connection)
    autodl.main(["code", "--env", str(tmp_path / "missing.json"), "--dry-run"])
    assert capsys.readouterr().out == "config.json\npyproject.toml\n2 files -> /srv/shared/nanochat\n"


@pytest.mark.parametrize("path", ["relative/path", "/", "/srv/../nanochat"])
def test_invalid_project_path_is_rejected_before_connecting(tmp_path, path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"platforms": {"AutoDL": {"project_dir": path}}}))
    with pytest.raises(ValueError, match="absolute project directory"):
        autodl.main(["code", "--config", str(config_path)])


def test_entry_passes_login_and_path_to_upload(tmp_path, monkeypatch, capsys):
    local = tmp_path / "nanochat/train.py"
    local.parent.mkdir()
    local.write_text("print('train')")
    login = {"host": "example.invalid", "port": 2222, "user": "root", "passwd": "test-password"}
    project_dir = "/srv/custom/nanochat"
    env_path = tmp_path / "env.json"
    env_path.write_text(json.dumps({"AutoDL": {"ssh": login}}))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"platforms": {"AutoDL": {"project_dir": project_dir}}}))
    client = MagicMock()
    client.__enter__.return_value = client
    sftp = client.open_sftp.return_value.__enter__.return_value

    def stat(path):
        if path != "/":
            raise FileNotFoundError(path)

    sftp.stat.side_effect = stat
    monkeypatch.setattr(remote.paramiko, "SSHClient", lambda: client)
    monkeypatch.setattr(autodl, "ROOT", tmp_path)
    autodl.main(["code", "--env", str(env_path)])
    client.load_system_host_keys.assert_called_once()
    connection = client.connect.call_args.kwargs
    assert (connection["hostname"], connection["port"], connection["username"], connection["password"]) == (
        login["host"], login["port"], login["user"], login["passwd"])
    destination = str(PurePosixPath(project_dir) / "nanochat/train.py")
    sftp.put.assert_has_calls([
        call(str(config_path), f"{project_dir}/config.json"),
        call(str(local), destination),
    ])
    sftp.chmod.assert_any_call(destination, local.stat().st_mode & 0o777)
    assert "test-password" not in capsys.readouterr().out


def test_data_checks_shared_mount_before_upload(tmp_path, monkeypatch):
    archive = tmp_path / "data.tar.zst"
    archive.write_bytes(b"dataset")
    settings = {"storage": {"archive_path": archive.name}, "platforms": {"AutoDL": {
        "shared_dir": "/shared", "archive_path": "/shared/data.tar.zst"}}}
    (tmp_path / "config.json").write_text(json.dumps(settings))
    (tmp_path / "env.json").write_text(json.dumps({"AutoDL": {"ssh": {}}}))
    client = MagicMock()
    client.__enter__.return_value = client
    monkeypatch.setattr(platform_cli, "connect", lambda **kwargs: client)
    monkeypatch.setattr(autodl, "ROOT", tmp_path)
    operations = []
    monkeypatch.setattr(platform_cli, "require_shared_storage", lambda ssh, path: operations.append(path))
    monkeypatch.setattr(platform_cli, "upload_data", lambda ssh, **kwargs: operations.append(kwargs))
    autodl.main(["data"])
    assert operations == [PurePosixPath("/shared"), {
        "local_archive": archive, "remote_archive": PurePosixPath("/shared/data.tar.zst")}]
