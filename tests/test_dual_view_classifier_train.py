# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
import copy
import json
import signal
import sys
from pathlib import Path

import pytest
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dual_view_classifier_train as train
import dual_view_training_data as data


class SpyClassifier(torch.nn.Module):
    def __init__(self, classes=4, stop=None):
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, classes))
        self.calls = []
        self.stop = stop

    def forward(self, x):
        self.calls.append(len(x))
        if self.stop is not None and self.training:
            self.stop.signal_number = signal.SIGINT
        return self.backbone(x)


def setup_engine(tmp_path, epochs=2):
    bundle = data.synthetic_bundle(tmp_path / "data", num_classes=4)
    transform = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    settings = {"max_views": 3, "epochs": epochs, "num_workers": 0, "learning_rate": 0.001,
                "weight_decay": 0.05, "seed": 8, "alpha": 0.5, "amp": False, "patience": 0,
                "initialization": {"kind": "synthetic-test"}}
    contract = train.build_run_contract(bundle, settings, {"input_size": [3, 8, 8]},
                                        (transform, transform), {"device": "cpu", "git_commit": "test"})
    return bundle, transform, contract


def test_hand_computed_logits_arbitrary_mapping_and_full_only():
    # Parent 0 full=[8,0], crops=[0,4],[4,8]; parent 1 full=[2,6].
    logits = torch.tensor([[0., 4.], [2., 6.], [8., 0.], [4., 8.]], requires_grad=True)
    outputs = train.fuse_logits(logits, torch.tensor([0, 1, 0, 0]), torch.tensor([1, 0, 0, 1]), 2)
    assert torch.equal(outputs["crop"][0], torch.tensor([2., 6.]))
    assert torch.equal(outputs["fused"], torch.tensor([[5., 3.], [2., 6.]]))
    assert outputs["crop_counts"].tolist() == [2, 0]
    expected = torch.softmax(torch.tensor([[5., 3.], [2., 6.]]), dim=1)
    assert torch.equal(train.fused_probabilities(outputs), expected)
    assert not torch.allclose(expected[0], 0.5 * logits[2].softmax(0) +
                              0.25 * logits[0].softmax(0) + 0.25 * logits[3].softmax(0))
    with pytest.raises(ValueError, match="zero-box"):
        train.parent_loss(outputs, torch.tensor([0, 1]))


def test_exact_N_zero_for_all_alpha_values():
    full = torch.tensor([[1.123456789, -45., 99.]], dtype=torch.float64)
    for alpha in (0.0, 0.5, 1.0):
        result = train.fuse_logits(full, torch.tensor([0]), torch.tensor([0]), 1, alpha=alpha)
        assert torch.equal(result["fused"], full)


def test_single_model_call_shared_gradients_and_parent_mean_loss(tmp_path):
    bundle, transform, _ = setup_engine(tmp_path)
    dataset = data.DualViewDataset(bundle, "train", transform)
    batch = data.collate_views([dataset[0], dataset[1]], max_views=5)
    batch["flat_views"].requires_grad_()
    model = SpyClassifier()
    outputs = train.forward_batch(model, batch, torch.device("cpu"))
    loss = train.parent_loss(outputs, batch["labels"])
    expected = sum(torch.nn.functional.cross_entropy(outputs["fused"][i:i+1], batch["labels"][i:i+1])
                   for i in range(2)) / 2
    assert torch.equal(loss, expected)
    loss.backward()
    assert model.calls == [5]
    assert all(batch["flat_views"].grad[i].abs().sum() > 0 for i in range(5))
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(name.startswith("backbone.") for name, _ in model.named_parameters())


def test_crop_gradient_is_exact_one_over_N_and_no_extra_parent_weight():
    logits = torch.tensor([[1., 2.], [3., 0.], [2., 1.], [0., 4.], [4., 0.]], requires_grad=True)
    outputs = train.fuse_logits(logits, torch.tensor([0, 0, 1, 1, 1]), torch.tensor([0, 1, 0, 1, 1]), 2)
    outputs["fused"].sum().backward()
    assert torch.equal(logits.grad, torch.tensor([[.5, .5], [.5, .5], [.5, .5], [.25, .25], [.25, .25]]))
    # Duplicating an identical crop cannot increase the parent's contribution.
    duplicate = train.fuse_logits(torch.tensor([[1., 2.], [3., 0.], [3., 0.]]),
                                  torch.tensor([0, 0, 0]), torch.tensor([0, 1, 1]), 1)
    assert torch.equal(duplicate["fused"][0], outputs["fused"][0])


@pytest.mark.parametrize("parents,roles,count", [([0, 0], [0, 0], 1), ([0, 0], [1, 1], 1),
                                                         ([0, 1], [0, 1], 1), ([-1, 0], [0, 1], 1),
                                                         ([0, 0], [0, 2], 1)])
def test_bad_mapping_rejected(parents, roles, count):
    with pytest.raises(ValueError):
        train.fuse_logits(torch.ones(2, 3), torch.tensor(parents), torch.tensor(roles), count)


def test_half_precision_crop_reduction_does_not_overflow():
    logits = torch.full((101, 2), 1000., dtype=torch.float16)
    result = train.fuse_logits(logits, torch.zeros(101, dtype=torch.long), torch.tensor([0] + [1] * 100), 1)
    assert torch.equal(result["fused"], torch.full((1, 2), 1000.))


@pytest.mark.parametrize("field", ["data_contract_sha256", "manifests", "class_map", "review_sources",
                                   "alpha", "crop_rule", "zero_rule", "aggregation", "max_views", "environment", "class_keys"])
def test_resume_rejects_every_bound_contract_change(tmp_path, field):
    bundle, transform, contract = setup_engine(tmp_path)
    payload = {"checkpoint_format": train.CHECKPOINT_FORMAT, "run_contract": copy.deepcopy(contract),
               "class_keys": contract["class_keys"], "status": "epoch_complete",
               "progress": {"epoch": 1, "next_batch": 0, "no_improvement": 0}}
    train.validate_resume(payload, contract)
    changed = copy.deepcopy(contract)
    if field in ("manifests", "class_map", "review_sources"):
        changed["data_contract"][field] = {}
    elif field == "alpha":
        changed["fusion"]["alpha"] = .4
    elif field == "zero_rule":
        changed["fusion"]["supervised_zero_box_rule"] = "full-only"
    elif field == "aggregation":
        changed["fusion"]["multi_box_aggregation"] = "sum"
    elif field == "max_views":
        changed["settings"]["max_views"] += 1
    else:
        changed[field] = "changed"
    with pytest.raises(ValueError, match="contract mismatch"):
        train.validate_resume(payload, changed)


class HeadFixture(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.features = torch.nn.Linear(2, 2)
        self.backbone.classifier = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(), torch.nn.Linear(2, classes))


def test_755_to_1224_exact_class_prefix_and_new_rows_preserved(tmp_path):
    old = HeadFixture(755)
    new = HeadFixture(1224)
    untouched_weight = new.state_dict()[train.CLASSIFIER_WEIGHT_KEY][755:].clone()
    untouched_bias = new.state_dict()[train.CLASSIFIER_BIAS_KEY][755:].clone()
    checkpoint = tmp_path / "old.pth"
    torch.save(old.state_dict(), checkpoint)
    class_map = tmp_path / "classes.json"
    classes = [{"model_class_id": i, "class_key": f"Exact_Key_{i}"} for i in range(1224)]
    class_map.write_bytes(data.canonical({"classes": classes[:755]}))
    keys = [c["class_key"] for c in classes]
    count = train.initialize_exact(new, checkpoint, data.digest(checkpoint.read_bytes()), class_map,
                                    data.digest(class_map.read_bytes()), keys)
    assert count == 755
    assert torch.equal(new.state_dict()[train.CLASSIFIER_WEIGHT_KEY][:755], old.state_dict()[train.CLASSIFIER_WEIGHT_KEY])
    assert torch.equal(new.state_dict()[train.CLASSIFIER_BIAS_KEY][:755], old.state_dict()[train.CLASSIFIER_BIAS_KEY])
    assert torch.equal(new.state_dict()[train.CLASSIFIER_WEIGHT_KEY][755:], untouched_weight)
    assert torch.equal(new.state_dict()[train.CLASSIFIER_BIAS_KEY][755:], untouched_bias)
    assert torch.equal(new.backbone.features.weight, old.backbone.features.weight)
    keys[0] = "exact_key_0"
    with pytest.raises(ValueError, match="exact target prefix"):
        train.initialize_exact(new, checkpoint, data.digest(checkpoint.read_bytes()), class_map,
                               data.digest(class_map.read_bytes()), keys)


def test_missing_backbone_key_rejected_before_model_mutation(tmp_path):
    old, model = HeadFixture(2), HeadFixture(3)
    state = old.state_dict()
    del state["backbone.features.bias"]
    path = tmp_path / "missing.pth"
    torch.save(state, path)
    class_map = tmp_path / "classes.json"
    class_map.write_bytes(data.canonical({"classes": [{"model_class_id": 0, "class_key": "a"}, {"model_class_id": 1, "class_key": "b"}]}))
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="exactly"):
        train.initialize_exact(model, path, data.digest(path.read_bytes()), class_map, data.digest(class_map.read_bytes()), ["a", "b", "c"])
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_interrupt_resume_equals_uninterrupted_optimizer_updates(tmp_path):
    bundle, transform, contract = setup_engine(tmp_path)
    train.seed_everything(101)
    reference = SpyClassifier()
    initial = copy.deepcopy(reference.state_dict())
    assert train.run_epochs(bundle, reference, (transform, transform), contract, tmp_path / "reference") == 0
    stop = train.StopRequest()
    interrupted = SpyClassifier(stop=stop)
    interrupted.load_state_dict(initial)
    run = tmp_path / "resumed"
    assert train.run_epochs(bundle, interrupted, (transform, transform), contract, run, stop_request=stop) == 130
    checkpoint = train.load_torch_file(run / "interrupted.checkpoint.pth", torch.device("cpu"))
    assert checkpoint["progress"]["next_batch"] == 1
    assert checkpoint["progress"]["epoch"] == 0
    assert checkpoint["optimizer_state_dict"]["state"]
    resumed = SpyClassifier()
    assert train.run_epochs(bundle, resumed, (transform, transform), contract, run,
                            resume_path=run / "interrupted.checkpoint.pth") == 0
    assert all(torch.equal(a, b) for a, b in zip(reference.parameters(), resumed.parameters(), strict=True))
    # Two train forwards + one validation forward each epoch, first train call
    # already happened before the interrupt and is NOT repeated on resume.
    assert len(reference.calls) == len(interrupted.calls) + len(resumed.calls) == 6
    final = train.load_torch_file(run / "latest.checkpoint.pth", torch.device("cpu"))
    expected = train.load_torch_file(tmp_path / "reference" / "latest.checkpoint.pth", torch.device("cpu"))
    assert final["scheduler_state_dict"] == expected["scheduler_state_dict"]
    for key in final["optimizer_state_dict"]["state"]:
        for name, tensor in final["optimizer_state_dict"]["state"][key].items():
            assert torch.equal(tensor, expected["optimizer_state_dict"]["state"][key][name])
    report = json.loads((run / "epoch-0001.json").read_text())
    assert report["train"]["parent_count"] == 2
    assert report["train"]["class_sample_counts"] == [1, 1, 0, 0]
    assert report["zero_box_exclusions"]["zero_box_exclusions"] == 1
    assert report["peak_cuda_memory_bytes"] is None


def test_new_run_never_overwrites_existing_directory(tmp_path):
    bundle, transform, contract = setup_engine(tmp_path)
    run = tmp_path / "existing"
    run.mkdir()
    marker = run / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        train.run_epochs(bundle, SpyClassifier(), (transform, transform), contract, run)
    assert marker.read_text() == "keep"


def test_sigterm_in_validation_resumes_without_repeating_training(tmp_path):
    bundle, transform, contract = setup_engine(tmp_path, epochs=1)
    stop = train.StopRequest()
    class ValidationStop(SpyClassifier):
        def forward(self, x):
            if not self.training:
                stop.signal_number = signal.SIGTERM
            return super().forward(x)
    model = ValidationStop()
    # Signal during the LAST validation forward must also be honored.
    run = tmp_path / "validation-interrupt"
    assert train.run_epochs(bundle, model, (transform, transform), contract, run, stop_request=stop) == 143
    checkpoint = train.load_torch_file(run / "interrupted.checkpoint.pth", torch.device("cpu"))
    assert checkpoint["progress"]["next_batch"] == 2
    restored = SpyClassifier()
    assert train.run_epochs(bundle, restored, (transform, transform), contract, run,
                            resume_path=run / "interrupted.checkpoint.pth") == 0
    assert restored.calls == [2]


def test_terminal_early_stopping_checkpoint_stays_terminal(tmp_path):
    bundle, transform, contract = setup_engine(tmp_path, epochs=4)
    contract["settings"]["patience"] = 1
    class ConstantPrediction(SpyClassifier):
        def forward(self, x):
            logits = super().forward(x)
            return logits * 0  # Differentiable, fixed prediction across epochs.
    run = tmp_path / "early-stop"
    assert train.run_epochs(bundle, ConstantPrediction(), (transform, transform), contract, run) == 0
    checkpoint = train.load_torch_file(run / "latest.checkpoint.pth", torch.device("cpu"))
    assert checkpoint["progress"]["epoch"] == 2
    model = ConstantPrediction()
    assert train.run_epochs(bundle, model, (transform, transform), contract, run,
                            resume_path=run / "latest.checkpoint.pth") == 0
    assert model.calls == []


def test_cli_refuses_unfrozen_contract_and_final_test_options(tmp_path):
    bundle, _, _ = setup_engine(tmp_path)
    assert train.main(["--project-root", str(bundle.root), "--data-contract", str(bundle.root / "data_contract.json"),
                       "--data-contract-sha256", bundle.contract_sha256, "--max-views", "5", "--allow-training"]) == 1
    with pytest.raises(SystemExit):
        train.parse_args(["--split", "final_test"])
    assert train.main(["--synthetic-smoke", "--allow-training"]) == 1


def test_real_convnext_cpu_forward_only_no_weight_download(monkeypatch):
    # One real shared ConvNeXt-Tiny; generated pixels only, no production weight.
    def no_download(*args, **kwargs):
        pytest.fail("tests must not download pretrained weights")
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", no_download)
    result = train.synthetic_smoke()
    assert result["input_shape"] == [5, 3, 224, 224]
    assert result["fused_shape"] == [2, 1224]
    assert result["forward_calls"] == 1
    assert result["crop_counts"] == [1, 2]
    assert result["source_hashes_unchanged"]


def test_exported_sources_keep_exact_commit_without_git(tmp_path, monkeypatch):
    source = tmp_path / "bilinear-cnn"
    source.mkdir()
    names = (
        "dual_view_training_data.py",
        "dual_view_classifier_train.py",
        "training_runtime.py",
    )
    for name in names:
        (source / name).write_bytes(b"# temporary constructed source\n")
    hashes = {name: data.digest((source / name).read_bytes()) for name in names}
    snapshot = {"format": "birdsvision-dual-view-source-snapshot-v1",
                "git_commit": "a" * 40, "source_sha256": hashes}
    (tmp_path / "source_snapshot.json").write_bytes(data.canonical(snapshot))
    monkeypatch.setattr(train.subprocess, "check_output", lambda *a, **kw: pytest.fail("archive must not require Git"))
    assert train.source_identity(source) == ("a" * 40, hashes)
    (source / "training_runtime.py").write_bytes(b"# changed\n")
    with pytest.raises(ValueError, match="source snapshot SHA-256 mismatch"):
        train.source_identity(source)


def test_exported_sources_reject_invalid_revision(tmp_path):
    source = tmp_path / "bilinear-cnn"
    source.mkdir()
    for name in (
        "dual_view_training_data.py",
        "dual_view_classifier_train.py",
        "training_runtime.py",
    ):
        (source / name).write_bytes(b"test")
    snapshot = {"format": "birdsvision-dual-view-source-snapshot-v1", "git_commit": "unverified"}
    (tmp_path / "source_snapshot.json").write_bytes(data.canonical(snapshot))
    with pytest.raises(ValueError, match="git_commit"):
        train.source_identity(source)


def config_values(tmp_path):
    return {
        "allow_training": False,
        "project_root": str(tmp_path),
        "data_contract": str(tmp_path / "data_contract.json"),
        "data_contract_sha256": "a" * 64,
        "output_dir": str(tmp_path / "run"),
        "device": "cuda",
        "max_views": 64,
        "epochs": 25,
        "num_workers": 4,
        "learning_rate": 0.0001,
        "weight_decay": 0.05,
        "seed": 20260907,
        "alpha": 0.5,
        "amp": True,
        "patience": 5,
        "max_runtime_minutes": 0,
        "resume": "auto",
        "init_checkpoint": str(tmp_path / "old.pth"),
        "init_checkpoint_sha256": "b" * 64,
        "init_class_map": str(tmp_path / "old_classes.json"),
        "init_class_map_sha256": "c" * 64,
    }


def test_config_only_mode_loads_exact_dictionary(tmp_path):
    path = tmp_path / "config.py"
    path.write_text("DUAL_VIEW_TRAINING_RUN = " + repr(config_values(tmp_path)), encoding="utf-8")
    args = train.parse_args(["--config", str(path)])
    assert args.max_views == 64
    assert args.resume == "auto"
    assert args.project_root == tmp_path.resolve()
    with pytest.raises(SystemExit):
        train.parse_args(["--config", str(path), "--epochs", "2"])


def test_config_only_mode_rejects_unknown_or_missing_fields(tmp_path):
    values = config_values(tmp_path)
    values["unexpected"] = 1
    del values["alpha"]
    path = tmp_path / "config.py"
    path.write_text("DUAL_VIEW_TRAINING_RUN = " + repr(values), encoding="utf-8")
    with pytest.raises(ValueError, match="missing=.*alpha.*unknown=.*unexpected"):
        train.load_training_config(path)


def test_auto_resume_selects_most_advanced_checkpoint(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    base = {"checkpoint_format": train.CHECKPOINT_FORMAT, "progress": {"epoch": 2, "next_batch": 0}}
    torch.save(base, run / "latest.checkpoint.pth")
    interrupted = copy.deepcopy(base)
    interrupted["progress"] = {"epoch": 2, "next_batch": 7}
    torch.save(interrupted, run / "interrupted.checkpoint.pth")
    assert train.resolve_auto_resume(run).name == "interrupted.checkpoint.pth"
    assert train.resolve_auto_resume(tmp_path / "new") is None


def test_species_group_loss_marginalizes_over_fine_outputs():
    outputs = {
        "fused": torch.tensor([[2.0, 1.0, -1.0], [0.0, 3.0, 1.0]]),
        "crop_counts": torch.tensor([1, 1]),
    }
    batch = {"target_class_ids": [[0, 1], [2]]}
    loss = train.parent_loss(outputs, batch)
    expected = torch.stack([
        torch.logsumexp(outputs["fused"][0], dim=0)
        - torch.logsumexp(outputs["fused"][0, [0, 1]], dim=0),
        torch.logsumexp(outputs["fused"][1], dim=0) - outputs["fused"][1, 2],
    ]).mean()
    assert torch.allclose(loss, expected)


def test_hierarchical_metric_scores_coarse_child_at_one_and_fine_sibling_at_point85():
    metrics = train.ParentMetrics(
        3, class_to_species_group=["duck", "duck", "other"]
    )
    box = {"bbox_xyxy_normalized": [0.1, 0.1, 0.6, 0.6]}
    records = [
        {"model_class_id": None, "supervision": {
            "species_catalog_entry_id": "duck", "model_class_ids": [0, 1],
        }, "instances": [box]},
        {"model_class_id": 0, "instances": [box]},
    ]
    batch = {
        "labels": torch.tensor([-1, 0]),
        "target_class_ids": [[0, 1], [0]],
        "records": records,
        "flat_views": torch.zeros(4, 3, 2, 2),
    }
    logits = torch.tensor([[0.0, 3.0, 1.0], [0.0, 3.0, 1.0]])
    outputs = {"full": logits, "crop": logits, "fused": logits}
    metrics.update(outputs, batch, 0.5)
    overall = metrics.report()["groups"]["overall"]
    assert overall["fused_top1"] == 50.0
    assert overall["fused_hierarchical_top1"] == 92.5
