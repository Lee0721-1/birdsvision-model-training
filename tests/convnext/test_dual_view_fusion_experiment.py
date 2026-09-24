# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from convnext import dual_view_fusion_experiment as experiment
from convnext import dual_view_classifier_finetune as fine_tune


def test_invocation_binds_exact_entrypoint_and_active_config(tmp_path):
    config = tmp_path / "exact_config.py"
    config.write_bytes(b"DUAL_VIEW_FINE_TUNING_RUN = {}\n")
    identity = experiment.invocation_identity(config)
    assert identity["entrypoint_name"] == "dual_view_fusion_experiment.py"
    assert identity["active_config_name"] == "exact_config.py"
    assert len(identity["entrypoint_sha256"]) == 64
    original = identity["active_config_sha256"]
    config.write_bytes(b"DUAL_VIEW_FINE_TUNING_RUN = {'changed': True}\n")
    assert experiment.invocation_identity(config)["active_config_sha256"] != original


def test_missing_active_config_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        experiment.invocation_identity(tmp_path / "missing.py")


def test_launcher_injects_invocation_into_runtime_contract_and_restores_hook(
        tmp_path, monkeypatch):
    config = tmp_path / "config.py"
    config.write_bytes(b"DUAL_VIEW_FINE_TUNING_RUN = {}\n")
    original = lambda device: {"device": device}
    observed = {}

    def delegated_main(argv):
        observed.update(experiment.train.runtime_contract("cuda"))
        observed["argv"] = argv
        return 17

    monkeypatch.setattr(experiment.train, "runtime_contract", original)
    monkeypatch.setattr(experiment.train, "main", delegated_main)
    assert experiment.main(["--config", str(config)]) == 17
    assert observed["device"] == "cuda"
    assert observed["experiment_invocation"]["active_config_name"] == "config.py"
    assert observed["argv"] == ["--config", str(config)]
    assert experiment.train.runtime_contract is original


def test_second_seed_config_changes_only_seed_and_output_directory():
    root = Path(__file__).resolve().parents[2]
    first = vars(fine_tune.load_training_config(
        root / "convnext" / "config_dual_view_fusion_v2.py"
    ))
    second = vars(fine_tune.load_training_config(
        root / "convnext" / "config_dual_view_fusion_v2_seed_20260909.py"
    ))
    ignored = {"config", "seed", "output_dir"}
    assert {key: value for key, value in first.items() if key not in ignored} == {
        key: value for key, value in second.items() if key not in ignored
    }
    assert first["seed"] == 20260908
    assert second["seed"] == 20260909
    assert first["output_dir"] != second["output_dir"]


def test_aux_010_config_changes_only_auxiliary_weights_and_output_directory():
    root = Path(__file__).resolve().parents[2]
    baseline = vars(fine_tune.load_training_config(
        root / "convnext" / "config_dual_view_fusion_v2.py"
    ))
    experiment_config = vars(fine_tune.load_training_config(
        root / "convnext" / "config_dual_view_fusion_v2_aux_010.py"
    ))
    ignored = {
        "config", "output_dir", "full_aux_loss_weight", "crop_aux_loss_weight",
    }
    assert {key: value for key, value in baseline.items() if key not in ignored} == {
        key: value for key, value in experiment_config.items() if key not in ignored
    }
    assert baseline["full_aux_loss_weight"] == 0.25
    assert baseline["crop_aux_loss_weight"] == 0.25
    assert experiment_config["full_aux_loss_weight"] == 0.10
    assert experiment_config["crop_aux_loss_weight"] == 0.10
    assert experiment_config["seed"] == 20260908
    assert baseline["output_dir"] != experiment_config["output_dir"]
