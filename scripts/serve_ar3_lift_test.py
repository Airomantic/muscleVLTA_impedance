import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def checkpoint(tmp_path):
    checkpoint = tmp_path / "20000"
    (checkpoint / "params/d").mkdir(parents=True)
    stats = checkpoint / "assets/ar3/norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text("{}")
    for name in ("_METADATA", "manifest.ocdbt", "d/weights"):
        (checkpoint / "params" / name).write_text("test data")
    tokenizer = tmp_path / "cache/big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("test tokenizer")
    return checkpoint


def serve(checkpoint, **overrides):
    env = {key: value for key, value in os.environ.items() if not key.startswith("OPENPI_")}
    env.update(
        OPENPI_DRY_RUN="1",
        OPENPI_PYTHON=sys.executable,
        OPENPI_DATA_HOME=str(checkpoint.parent / "cache"),
        **overrides,
    )
    return subprocess.run(
        ["bash", str(Path(__file__).with_name("serve_ar3_lift.sh")), str(checkpoint)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("missing", ["_METADATA", "manifest.ocdbt", "d/weights"])
def test_incomplete_checkpoint_fails_before_model_start(checkpoint, missing):
    (checkpoint / "params" / missing).unlink()
    result = serve(checkpoint)
    assert result.returncode == 1
    assert "Incomplete checkpoint" in result.stderr
    assert "Would run:" not in result.stdout


def test_complete_checkpoint_selects_its_normalization_stats(checkpoint, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/unrelated/python3.12/ros")
    result = serve(checkpoint)
    assert result.returncode == 0, result.stderr
    assert "OPENPI_AR3_ASSET_ID=ar3" in result.stdout
    assert "--checkpoint-dir" in result.stdout
    assert str(checkpoint) in result.stdout
    assert "/unrelated/python3.12/ros" not in result.stdout


def test_missing_selected_normalization_stats_fail_before_model_start(checkpoint):
    result = serve(checkpoint, OPENPI_AR3_ASSET_ID="wrong-dataset")
    assert result.returncode == 1
    assert "Missing normalization stats" in result.stderr
