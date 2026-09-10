# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dual_view_parameter_sweep as sweep
import dual_view_training_data as data


@pytest.mark.parametrize("key,value", [
    ("epochs", 0), ("epochs", True), ("trainable", "unknown"),
    ("learning_rates", [float("nan")]), ("learning_rates", [0]),
    ("weight_decays", [-1]), ("seeds", [True]), ("seeds", [1, 1]),
    ("validation_per_class", -1), ("subset_seed", -1), ("format", "other"),
])
def test_invalid_plan(key, value):
    plan = copy.deepcopy(sweep.EXAMPLE)
    plan[key] = value
    with pytest.raises(ValueError):
        sweep.validate_plan(plan)


def test_unknown_fields_fail_instead_of_guessing():
    with pytest.raises(ValueError):
        sweep.validate_plan({**sweep.EXAMPLE, "LearningRate": 0.1})


def test_shipped_example_matches_generator():
    assert sweep.read(Path(sweep.__file__).with_name("dual_view_sweep_example.json")) == sweep.EXAMPLE


def test_supervisor_resume_skips_completed_trials(tmp_path, monkeypatch):
    calls = []
    def fake_worker(command, directory, args):
        stage = command[command.index("--internal-stage") + 1]
        calls.append(stage)
        report = {"exit_code": 0, "stop_reason": None, "peak_observed_hwm_bytes": 100,
                  "phase_peak_sampled_rss_bytes": {}, "elapsed_seconds": 0.1}
        if stage == "prepare":
            sweep.save(args.output_dir / "sweep_contract.json", {"fixture": True})
        else:
            plan = {**sweep.EXAMPLE, "epochs": 1, "learning_rates": [0.0001, 0.0003],
                    "weight_decays": [0.05], "seeds": [20260907]}
            trial = next(t for t in sweep.trials(plan) if t["trial_id"] == directory.name)
            (directory / "training").mkdir()
            path = directory / "training" / "latest.checkpoint.pth"
            path.write_bytes(b"temporary synthetic checkpoint")
            sweep.save(directory / "result.json", {**result_for(trial),
                "sweep_contract_sha256": sweep.sha(args.output_dir / "sweep_contract.json"),
                "checkpoint": "training/latest.checkpoint.pth", "checkpoint_sha256": sweep.sha(path)})
            sweep.save(directory / "resources-001.json", report)
        return report
    monkeypatch.setattr(sweep, "monitored", fake_worker)
    arguments = ["--synthetic-smoke", "--output-dir", str(tmp_path / "run")]
    assert sweep.main(arguments) == 0
    assert calls == ["prepare", "trial", "trial"]
    assert sweep.main([*arguments, "--resume-sweep"]) == 0
    assert calls == ["prepare", "trial", "trial", "prepare"]
    assert sweep.read(tmp_path / "run" / "comparison.json")["complete"]


def test_stratified_subset_fixed_across_trials_and_whole_parents(tmp_path):
    bundle = data.synthetic_bundle(tmp_path / "data", num_classes=1)
    before = copy.deepcopy(bundle.records)
    plan = {**sweep.EXAMPLE, "train_per_class": 1}
    selected = sweep.select_records(bundle.records, plan)
    assert len(selected) == 2
    assert selected == sweep.select_records(bundle.records, {**plan, "learning_rates": [0.5]})
    assert bundle.records == before
    assert all(row["instances"] == next(r["instances"] for r in before if r["record_id"] == row["record_id"]) for row in selected)
    forbidden = copy.deepcopy(before)
    forbidden[0]["split"] = "final_test"
    with pytest.raises(ValueError, match="sealed"):
        sweep.select_records(forbidden, plan)


class SmallSharedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.features = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.9))
        self.backbone.classifier = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(), torch.nn.Linear(4, 2))
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return self.backbone.classifier(self.backbone.features(images))


def test_head_mode_freezes_features_keeps_state_keys_and_one_forward():
    model = SmallSharedModel()
    keys = set(model.state_dict())
    sweep.configure_trainable(model, "head")
    model.train()
    model(torch.randn(5, 3)).sum().backward()
    assert model.calls == 1
    assert set(model.state_dict()) == keys
    assert not model.backbone.features.training
    assert model.backbone.classifier[2].training
    assert all(p.grad is None for p in model.backbone.features.parameters())
    assert all(p.grad is not None for p in model.backbone.classifier[2].parameters())


def test_all_mode_retains_feature_gradients():
    model = SmallSharedModel()
    sweep.configure_trainable(model, "all")
    model(torch.randn(5, 3)).sum().backward()
    assert all(p.requires_grad for p in model.parameters())
    assert model.backbone.features[0].weight.grad is not None


def test_confidence_and_calibration_are_parent_level_after_fusion(tmp_path):
    class Identity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x
    model = Identity()
    batch = {"flat_views": torch.tensor([[2., 0.], [2., 0.], [2., 0.], [2., 0.], [2., 0.]]),
             "view_parent": torch.tensor([0, 0, 1, 1, 1]), "view_role": torch.tensor([0, 1, 0, 1, 1]),
             "labels": torch.tensor([0, 1]), "records": [{"record_id": "p0"}, {"record_id": "p1"}]}
    result = sweep.evaluate(model, [batch], tmp_path / "predictions.jsonl")
    confidence = float(torch.tensor([2., 0.], dtype=torch.float64).softmax(0)[0])
    assert model.calls == 1
    assert result["parents"] == 2
    assert result["top1"] == result["macro_top1"] == 0.5
    assert result["mean_confidence"] == pytest.approx(confidence)
    assert result["wrong_mean_confidence"] == pytest.approx(confidence)
    assert result["ece"] == pytest.approx(confidence - 0.5)
    assert result["nll"] == pytest.approx(float(torch.nn.functional.cross_entropy(
        torch.tensor([[2., 0.], [2., 0.]], dtype=torch.float64), torch.tensor([0, 1]))))
    assert len((tmp_path / "predictions.jsonl").read_text().splitlines()) == 2


def result_for(trial, accuracy=0.5, confidence=0.9, nll=1.):
    return {"trial": trial, "validation": {"macro_top1": accuracy, "top1": accuracy, "top3": 1.,
            "mean_confidence": confidence, "nll": nll, "ece": abs(confidence - accuracy), "brier": 0.5}}


def test_rank_prefers_correctness_not_high_confidence_and_averages_seeds():
    plan = {**sweep.EXAMPLE, "learning_rates": [0.001, 0.01], "weight_decays": [0.05]}
    entries = sweep.trials(plan)
    results = [result_for(t, 0.9 if t["group_id"] == "group-0001" else 0.2,
                          0.9 if t["group_id"] == "group-0001" else 0.999) for t in entries]
    ranked = sweep.rank_results(plan, results, synthetic=False)
    assert ranked["best_parameters"]["learning_rate"] == 0.001
    assert ranked["best_parameters"]["seeds"] == plan["seeds"]
    assert sweep.rank_results(plan, results[:-1], synthetic=False)["best_parameters"] is None
    assert sweep.rank_results(plan, results, synthetic=True)["best_parameters"] is None


def test_nll_breaks_accuracy_tie():
    plan = {**sweep.EXAMPLE, "learning_rates": [0.001, 0.01], "weight_decays": [0.05], "seeds": [1]}
    entries = sweep.trials(plan)
    result = sweep.rank_results(plan, [result_for(entries[0], nll=2), result_for(entries[1], nll=1)], synthetic=False)
    assert result["best_parameters"]["learning_rate"] == 0.01


def monitor_args():
    return SimpleNamespace(threads=1, max_rss_mib=1024, min_available_mib=768,
                           min_free_disk_mib=1, sample_seconds=0.1, max_trial_minutes=0)


def test_monitored_child_records_memory_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "memory_sample", lambda pid: {
        "rss_bytes": 100, "hwm_bytes": 200, "available_bytes": 2 * 1024**3})
    report = sweep.monitored([sys.executable, "-c", "import time; print('child'); time.sleep(.3)"], tmp_path, monitor_args())
    assert report["exit_code"] == 0
    assert report["peak_sampled_rss_bytes"] == 100
    assert report["peak_observed_hwm_bytes"] == 200
    assert report["phase_peak_sampled_rss_bytes"]["worker_start"] == 100
    assert "child" in (tmp_path / report["log"]).read_text()
    assert report["memory_samples"] > 0
    assert not report["hard_memory_limit"]


def test_memory_guard_stops_child_and_preserves_report(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "memory_sample", lambda pid: {
        "rss_bytes": 2 * 1024**3, "hwm_bytes": 2 * 1024**3, "available_bytes": 2 * 1024**3})
    report = sweep.monitored([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, monitor_args())
    assert report["stop_reason"] == "worker_rss_limit"
    assert report["exit_code"] != 0
    assert list(tmp_path.glob("resources-*.json"))


def test_low_host_memory_refuses_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "memory_sample", lambda pid: {"rss_bytes": 1, "hwm_bytes": 1, "available_bytes": 1})
    report = sweep.monitored(["must-not-run"], tmp_path, monitor_args())
    assert report["stop_reason"] == "host_available_memory_floor_before_launch"
    assert report["exit_code"] == 125
    assert report["log"] is None
    assert list(tmp_path.glob("resources-*.json"))


def test_prepare_resume_binds_plan_sources_and_data(tmp_path):
    args = sweep.parser().parse_args(["--synthetic-smoke", "--output-dir", str(tmp_path), "--max-views", "3"])
    plan = {**sweep.EXAMPLE, "epochs": 1}
    bundle, contract = sweep.prepare(args, plan)
    assert len(bundle.records) == 3
    assert sweep.prepare(args, plan)[1] == contract
    with pytest.raises(ValueError, match="resume contract mismatch"):
        sweep.prepare(args, {**plan, "epochs": 2})


def test_real_cli_never_accepts_synthetic_contract(tmp_path):
    bundle = data.synthetic_bundle(tmp_path / "data")
    args = sweep.parser().parse_args(["--output-dir", str(tmp_path), "--max-views", "3",
        "--project-root", str(bundle.root), "--data-contract", str(bundle.root / "data_contract.json"),
        "--data-contract-sha256", bundle.contract_sha256])
    with pytest.raises(ValueError, match="frozen_user_confirmed"):
        sweep.prepare(args, sweep.EXAMPLE)


def test_final_test_declaration_rejected_before_referenced_files(tmp_path):
    path = tmp_path / "contract.json"
    document = {"format": data.FORMAT, "manifests": {"final_test": {"storage_path": "must-not-open"}}}
    path.write_bytes(data.canonical(document))
    args = sweep.parser().parse_args(["--output-dir", str(tmp_path), "--max-views", "3",
        "--project-root", str(tmp_path), "--data-contract", str(path), "--data-contract-sha256", sweep.sha(path)])
    with pytest.raises(ValueError, match="sealed"):
        sweep.prepare(args, sweep.EXAMPLE)


def test_synthetic_cli_cannot_take_real_inputs(tmp_path):
    assert sweep.main(["--synthetic-smoke", "--output-dir", str(tmp_path), "--project-root", "anything"]) == 1
    with pytest.raises(SystemExit):
        sweep.parser().parse_args(["--split", "final_test"])


def test_preflight_subprocess_failure_stops_before_trials(tmp_path, monkeypatch):
    calls = []
    def fail(command, directory, args):
        calls.append(command)
        return {"exit_code": 1, "stop_reason": "worker_rss_limit"}
    monkeypatch.setattr(sweep, "monitored", fail)
    assert sweep.main(["--synthetic-smoke", "--output-dir", str(tmp_path / "new")]) == 1
    assert len(calls) == 1
    assert not (tmp_path / "new" / "sweep.lock").exists()


def test_supervisor_import_does_not_load_torch():
    code = "import sys; import dual_view_parameter_sweep; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=Path(sweep.__file__).parent, check=True)


def test_collect_records_failed_attempt_and_refuses_tampered_checkpoint(tmp_path):
    plan = {**sweep.EXAMPLE, "learning_rates": [0.001], "weight_decays": [0.05], "seeds": [1]}
    args = SimpleNamespace(output_dir=tmp_path, synthetic_smoke=False)
    sweep.save(tmp_path / "sweep_contract.json", {"test": "constructed"})
    trial = sweep.trials(plan)[0]
    directory = tmp_path / trial["trial_id"]
    (directory / "training").mkdir(parents=True)
    checkpoint = directory / "training" / "latest.checkpoint.pth"
    checkpoint.write_bytes(b"constructed checkpoint")
    result = {**result_for(trial), "sweep_contract_sha256": sweep.sha(tmp_path / "sweep_contract.json"),
              "checkpoint": "training/latest.checkpoint.pth", "checkpoint_sha256": sweep.sha(checkpoint)}
    sweep.save(directory / "result.json", result)
    report = {"exit_code": 143, "stop_reason": "worker_rss_limit", "peak_observed_hwm_bytes": 100,
              "phase_peak_sampled_rss_bytes": {"training_with_epoch_validation_and_checkpoints": 90}, "elapsed_seconds": 1.}
    sweep.save(directory / "resources-001.json", report)
    summary = sweep.collect(args, plan)
    assert summary["best_parameters"] is None
    assert summary["trials"][0]["status"] == "unfinished"
    assert summary["trials"][0]["training_peak_sampled_rss_bytes"] == 90
    assert "unfinished" in (tmp_path / "comparison.csv").read_text(encoding="utf-8-sig")
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        sweep.collect(args, plan)
