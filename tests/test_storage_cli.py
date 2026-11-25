import json
from pathlib import Path

import pytest
import torch

from nanochat.__main__ import main
from nanochat.config import Config
from script.data import main as data_main


def test_project_config_loads_without_credentials():
    config = Config.load()
    assert config.model.n_layer == 24
    assert config.platforms["AutoDL"]["project_dir"] == "/root/nanochat"
    assert "passwd" not in json.dumps(config.to_dict())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cli_train_inspect_and_generate(tiny_config, tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(tiny_config.to_dict()))
    main(["inspect", "--config", str(path)])
    assert json.loads(capsys.readouterr().out)["recipe"]["grad_accum_steps"] == 2
    main(["train", "--config", str(path)])
    assert "saved final model" in capsys.readouterr().out
    main(["generate", "--config", str(path), "--prompt", "abc", "--max-tokens", "2"])
    assert "abc" in capsys.readouterr().out


def test_inspect_does_not_require_cuda(tiny_config, tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(tiny_config.to_dict()))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    main(["inspect", "--config", str(path)])
    assert json.loads(capsys.readouterr().out)["recipe"]["max_steps"] == 4


@pytest.mark.parametrize("command", [["train"], ["generate", "--prompt", "hello"]])
def test_training_commands_fail_without_cuda(command, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU is required"):
        main(command)


def test_training_data_dir_overrides_json_without_changing_it(tiny_config, tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(tiny_config.to_dict()))
    override = tmp_path / "custom tokens"
    observed = []
    monkeypatch.setattr("nanochat.train.train", lambda cfg: observed.append(cfg.storage.data_dir))
    main(["train", "--config", str(path), "--data-dir", str(override)])
    assert observed == [str(override)]
    assert Config.load(path).storage.data_dir == tiny_config.storage.data_dir


def test_pack_extract_cli_path_overrides(tiny_config, tmp_path, capsys):
    source = Path(tiny_config.storage.data_dir)
    archive = tmp_path / "custom archive.tar.zst"
    settings = tiny_config.to_dict()
    settings["storage"] = {"data_dir": str(tmp_path / "wrong data"),
                           "archive_path": str(tmp_path / "wrong.tar.zst")}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(settings))
    data_main(["pack", "--config", str(config_path), "--data-dir", str(source),
               "--archive-path", str(archive)])
    assert capsys.readouterr().out.strip() == str(archive)
    destination = tmp_path / "custom tokens"
    destination.mkdir()
    (destination / "stale.bin").write_bytes(b"old data")
    data_main(["extract", "--config", str(config_path), "--data-dir", str(destination),
               "--archive-path", str(archive)])
    assert capsys.readouterr().out.strip() == str(destination)
    assert {p.name for p in destination.iterdir()} == {p.name for p in source.iterdir()}
    for path in destination.iterdir():
        assert path.read_bytes() == (source / path.name).read_bytes()
    assert json.loads(config_path.read_text()) == settings


def test_unknown_config_section_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"modle": {}}))
    with pytest.raises(ValueError, match="unknown config sections"):
        Config.load(path)
