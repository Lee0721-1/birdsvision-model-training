# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
import copy
import json
import signal
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dual_view_classifier_finetune as fine
import dual_view_training_data as data


class SpyClassifier(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(classes, dtype=torch.float32))
        self.calls = []

    def forward(self, views):
        self.calls.append(len(views))
        signal = views.mean(dim=(1, 2, 3)).unsqueeze(1)
        return signal * self.weight.unsqueeze(0)


def batch_mapping(counts):
    parents, roles = [], []
    for parent, crop_count in enumerate(counts):
        parents.extend([parent] * (1 + crop_count))
        roles.extend([0] + [1] * crop_count)
    return torch.tensor(parents), torch.tensor(roles)


def test_adaptive_fusion_uses_point_four_then_point_two_five_and_full_only():
    parents, roles = batch_mapping([0, 2, 4])
    logits = torch.arange(len(parents) * 3, dtype=torch.float32).reshape(len(parents), 3)
    outputs = fine.adaptive_fuse_logits(
        logits, parents, roles, 3, default_alpha=0.4,
        many_box_threshold=3, many_box_alpha=0.25,
    )
    assert outputs["crop_counts"].tolist() == [0, 2, 4]
    assert outputs["alpha"].tolist() == pytest.approx([0.4, 0.4, 0.25])
    assert torch.equal(outputs["fused"][0], outputs["full"][0])
    assert torch.allclose(
        outputs["fused"][1], 0.4 * outputs["full"][1] + 0.6 * outputs["crop"][1]
    )
    assert torch.allclose(
        outputs["fused"][2], 0.25 * outputs["full"][2] + 0.75 * outputs["crop"][2]
    )


def test_forward_flattens_all_views_into_one_shared_model_call():
    model = SpyClassifier()
    parents, roles = batch_mapping([1, 4])
    batch = {
        "flat_views": torch.rand(len(parents), 3, 4, 4),
        "view_parent": parents, "view_role": roles,
        "labels": torch.tensor([0, 1]),
    }
    outputs = fine.forward_batch(model, batch, torch.device("cpu"), {
        "default_alpha": 0.4, "many_box_threshold": 3, "many_box_alpha": 0.25,
    })
    assert model.calls == [7]
    assert outputs["fused"].shape == (2, 4)


def test_auxiliary_loss_is_parent_mean_and_normalized_by_component_weights():
    outputs = {
        "fused": torch.tensor([[3.0, 0.0], [0.0, 2.0]]),
        "full": torch.tensor([[2.0, 0.0], [1.0, 0.0]]),
        "crop": torch.tensor([[1.0, 0.0], [0.0, 3.0]]),
        "crop_counts": torch.tensor([1, 5]),
    }
    batch = {"target_class_ids": [[0], [1]]}
    losses = fine.parent_losses(outputs, batch, full_weight=0.25, crop_weight=0.25)
    expected = (losses["fused"] + 0.25 * losses["full"] + 0.25 * losses["crop"]) / 1.5
    assert torch.allclose(losses["total"], expected)
    duplicated = copy.deepcopy(outputs)
    duplicated["crop_counts"] = torch.tensor([1, 50])
    assert torch.allclose(
        fine.parent_losses(duplicated, batch, full_weight=0.25, crop_weight=0.25)["total"],
        losses["total"],
    )


def config_values(tmp_path):
    return {
        "allow_training": False,
        "project_root": str(tmp_path),
        "data_contract": str(tmp_path / "data_contract.json"),
        "data_contract_sha256": "a" * 64,
        "output_dir": str(tmp_path / "run"),
        "device": "cuda", "max_views": 64, "epochs": 8,
        "num_workers": 4, "learning_rate": 1e-5, "weight_decay": 0.05,
        "seed": 20260908, "default_alpha": 0.4,
        "many_box_threshold": 3, "many_box_alpha": 0.25,
        "full_aux_loss_weight": 0.25, "crop_aux_loss_weight": 0.25,
        "tiny_crop_area_threshold": 0.01, "tiny_crop_scale_min": 0.5,
        "amp": True, "patience": 3, "max_runtime_minutes": 0,
        "resume": "auto", "init_checkpoint": str(tmp_path / "best.pth"),
        "init_checkpoint_sha256": "b" * 64,
    }


def test_config_is_exact_and_cli_has_no_final_test_route(tmp_path):
    path = tmp_path / "config.py"
    path.write_text(
        "DUAL_VIEW_FINE_TUNING_RUN = " + repr(config_values(tmp_path)),
        encoding="utf-8",
    )
    args = fine.parse_args(["--config", str(path)])
    assert args.default_alpha == 0.4
    assert args.many_box_alpha == 0.25
    with pytest.raises(SystemExit):
        fine.parse_args(["--split", "final_test"])
    values = config_values(tmp_path)
    values["unknown"] = 1
    del values["tiny_crop_scale_min"]
    path.write_text(
        "DUAL_VIEW_FINE_TUNING_RUN = " + repr(values), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="tiny_crop_scale_min.*unknown"):
        fine.load_training_config(path)


def setup_engine(tmp_path):
    bundle = data.synthetic_bundle(tmp_path / "data", num_classes=4)
    transform = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    settings = {
        "max_views": 5, "epochs": 1, "num_workers": 0,
        "learning_rate": 1e-3, "weight_decay": 0.0, "seed": 7,
        "default_alpha": 0.4, "many_box_threshold": 3,
        "many_box_alpha": 0.25, "full_aux_loss_weight": 0.25,
        "crop_aux_loss_weight": 0.25, "tiny_crop_area_threshold": 0.01,
        "tiny_crop_scale_min": 0.5, "amp": False, "patience": 0,
        "initialization": {"checkpoint_sha256": "a" * 64},
    }
    contract = {
        "format": fine.CHECKPOINT_FORMAT,
        "class_keys": [item["class_key"] for item in bundle.classes],
        "settings": settings,
        "environment": {"device": "cpu"},
    }
    return bundle, (transform, transform, transform), contract


def test_independent_engine_trains_and_records_component_losses(tmp_path):
    bundle, transform_set, contract = setup_engine(tmp_path)
    run = tmp_path / "run"
    model = SpyClassifier()
    assert fine.run_epochs(bundle, model, transform_set, contract, run) == 0
    report = json.loads((run / "epoch-0001.json").read_text(encoding="utf-8"))
    assert report["train"]["parent_count"] == 2
    assert set(report["train"]["loss_components"]) == {"total", "fused", "full", "crop"}
    checkpoint = fine.load_torch_file(run / "latest.checkpoint.pth", torch.device("cpu"))
    assert checkpoint["checkpoint_format"] == fine.CHECKPOINT_FORMAT


def test_auto_resume_config_keeps_original_initialization_binding(tmp_path):
    values = config_values(tmp_path)
    values.update(device="cpu", allow_training=True, epochs=1, max_views=5,
                  num_workers=0, amp=False)
    path = tmp_path / "config.py"
    path.write_text(
        "DUAL_VIEW_FINE_TUNING_RUN = " + repr(values), encoding="utf-8"
    )
    args = fine.load_training_config(path)
    assert args.resume == "auto"
    assert args.init_checkpoint_sha256 == "b" * 64


def test_completed_fine_tune_checkpoint_resumes_as_terminal(tmp_path):
    bundle, transform_set, contract = setup_engine(tmp_path)
    run = tmp_path / "run"
    assert fine.run_epochs(bundle, SpyClassifier(), transform_set, contract, run) == 0
    restored = SpyClassifier()
    assert fine.run_epochs(
        bundle, restored, transform_set, contract, run,
        resume_path=run / "latest.checkpoint.pth",
    ) == 0
    assert restored.calls == []


def test_initialization_rejects_wrong_format_and_binds_epoch_and_contract(tmp_path):
    model = SpyClassifier()
    class_keys = [f"class:{index}" for index in range(4)]
    payload = {
        "checkpoint_format": fine.BASE_CHECKPOINT_FORMAT,
        "status": "epoch_complete", "class_keys": class_keys,
        "progress": {"epoch": 22}, "run_contract": {"alpha": 0.5},
        "model_state_dict": model.state_dict(),
    }
    path = tmp_path / "best.pth"
    torch.save(payload, path)
    sha = data.digest(path.read_bytes())
    identity = fine.load_initial_checkpoint(SpyClassifier(), path, sha, class_keys)
    assert identity["completed_epoch"] == 22
    assert identity["checkpoint_sha256"] == sha
    payload["checkpoint_format"] = "wrong"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="completed dual-view baseline"):
        fine.load_initial_checkpoint(SpyClassifier(), path, data.digest(path.read_bytes()), class_keys)
