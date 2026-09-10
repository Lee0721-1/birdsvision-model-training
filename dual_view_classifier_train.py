#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Shared ConvNeXt-Tiny, one forward per logical batch, external logits fusion.

Default CLI is read-only preflight. --synthetic-smoke constructs temporary data
and performs CPU forward only. Training requires a separately frozen, explicitly
hash-bound data contract AND --allow-training. No final_test option exists.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import platform
import random
import runpy
import signal
import subprocess
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training_runtime import (
    CHECKPOINT_FORMAT as LEGACY_FORMAT,
    CLASSIFIER_BIAS_KEY, CLASSIFIER_WEIGHT_KEY, StableCUBModel, StopRequest,
    atomic_json_save, atomic_torch_save, load_torch_file, make_transforms,
    seed_everything,
)
from dual_view_training_data import (
    CROP_RULE, ZERO_RULE, DualViewDataset, ViewBudgetBatchSampler,
    canonical, collate_views, digest, load_bundle, relative_file,
    synthetic_bundle, verified_bytes,
)


CHECKPOINT_FORMAT = "birdsvision-dual-view-checkpoint-v1"
CONFIG_ATTRIBUTE = "DUAL_VIEW_TRAINING_RUN"
CONFIG_ARGUMENT = "--config"
CONFIG_KEYS = frozenset({
    "allow_training", "project_root", "data_contract", "data_contract_sha256",
    "output_dir", "device", "max_views", "epochs", "num_workers",
    "learning_rate", "weight_decay", "seed", "alpha", "amp", "patience",
    "max_runtime_minutes", "resume", "init_checkpoint",
    "init_checkpoint_sha256", "init_class_map", "init_class_map_sha256",
})
CONFIG_PATH_KEYS = frozenset({
    "project_root", "data_contract", "output_dir", "init_checkpoint", "init_class_map",
})
FUSION_RULE = {
    "location": "logits", "alpha": 0.5,
    "multi_box_aggregation": "equal_mean_crop_logits_1_over_N",
    "loss": "mean_parent_exact_or_species_marginal_nll_no_auxiliary_loss",
    "supervised_zero_box_rule": ZERO_RULE,
    "normal_inference_zero_box_rule": "exact_full_only",
    "softmax": "after_fusion_only",
}


def load_training_config(config_path):
    """Load one exact config.py dictionary without hidden CLI overrides."""
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"training config does not exist: {config_path}")
    raw = runpy.run_path(str(config_path)).get(CONFIG_ATTRIBUTE)
    if not isinstance(raw, dict):
        raise ValueError(f"{CONFIG_ATTRIBUTE} must be a dictionary")
    missing, unknown = CONFIG_KEYS - set(raw), set(raw) - CONFIG_KEYS
    if missing or unknown:
        raise ValueError(
            f"invalid {CONFIG_ATTRIBUTE} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    values = dict(raw)
    for key in CONFIG_PATH_KEYS:
        value = values[key]
        if value is None:
            if key not in ("output_dir", "init_checkpoint", "init_class_map"):
                raise ValueError(f"{key} cannot be None")
            continue
        if not isinstance(value, (str, Path)):
            raise ValueError(f"{key} must be a path")
        path = Path(value).expanduser()
        values[key] = (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
    resume = values["resume"]
    if resume != "auto" and resume is not None:
        if not isinstance(resume, (str, Path)):
            raise ValueError("resume must be None, 'auto', or a checkpoint path")
        path = Path(resume).expanduser()
        values["resume"] = (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
    for key in ("allow_training", "amp"):
        if type(values[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    return argparse.Namespace(synthetic_smoke=False, config=config_path, **values)


def resolve_auto_resume(output_dir):
    """Select the most advanced complete/interrupted checkpoint in one run."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    if not output_dir.is_dir():
        raise ValueError("configured output_dir exists but is not a directory")
    choices = []
    for name in ("latest.checkpoint.pth", "interrupted.checkpoint.pth"):
        path = output_dir / name
        if path.is_file():
            payload = load_torch_file(path, torch.device("cpu"))
            if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
                raise ValueError(f"auto-resume found an invalid checkpoint: {path}")
            progress = payload.get("progress", {})
            if any(type(progress.get(key)) is not int for key in ("epoch", "next_batch")):
                raise ValueError(f"auto-resume found invalid progress: {path}")
            choices.append(((progress["epoch"], progress["next_batch"]), path))
    if not choices:
        raise FileExistsError("configured output_dir exists without a resumable checkpoint")
    return max(choices, key=lambda value: value[0])[1]


def fuse_logits(flat_logits, view_parent, view_role, parent_count, *, alpha=0.5):
    """Roles: 0=full, 1=crop. Mapping order may be arbitrary. No model/softmax."""
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite in [0, 1]")
    if (flat_logits.ndim != 2 or not flat_logits.is_floating_point()
            or flat_logits.shape[1] == 0 or type(parent_count) is not int or parent_count <= 0):
        raise ValueError("invalid logits/parent_count")
    if (view_parent.shape != (flat_logits.shape[0],) or view_role.shape != view_parent.shape
            or view_parent.dtype != torch.long or view_role.dtype != torch.long):
        raise ValueError("invalid view mapping shape/dtype")
    parents, roles = view_parent.to(flat_logits.device), view_role.to(flat_logits.device)
    if (torch.any(parents < 0) or torch.any(parents >= parent_count)
            or torch.any((roles != 0) & (roles != 1))):
        raise ValueError("out-of-range parent or view role")
    full_mask, crop_mask = roles == 0, roles == 1
    full_counts = torch.bincount(parents[full_mask], minlength=parent_count)
    if not torch.all(full_counts == 1):
        raise ValueError("each parent requires exactly one full view")
    crop_counts = torch.bincount(parents[crop_mask], minlength=parent_count)
    # Accumulate low-precision logits in float32 to avoid fp16 overflow at large N.
    logits = flat_logits.float() if flat_logits.dtype in (torch.float16, torch.bfloat16) else flat_logits
    zeros = logits.new_zeros((parent_count, logits.shape[1]))
    full = zeros.index_add(0, parents[full_mask], logits[full_mask])
    crop_sum = zeros.index_add(0, parents[crop_mask], logits[crop_mask])
    crop_mean = crop_sum / crop_counts.clamp_min(1).unsqueeze(1)
    fused = torch.where((crop_counts > 0).unsqueeze(1),
                        alpha * full + (1 - alpha) * crop_mean, full)
    return {"full": full, "crop": crop_mean, "fused": fused, "crop_counts": crop_counts}


def forward_batch(classifier, batch, device, *, alpha=0.5, amp=False):
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        flat_logits = classifier(batch["flat_views"].to(device))
    return fuse_logits(flat_logits, batch["view_parent"], batch["view_role"],
                       len(batch["labels"]), alpha=alpha)


def parent_loss(outputs, targets):
    if torch.any(outputs["crop_counts"] == 0):
        raise ValueError("zero-box parents cannot receive supervised loss")
    logits = outputs["fused"]
    if isinstance(targets, torch.Tensor):
        return F.cross_entropy(logits, targets, reduction="mean")
    target_ids = targets["target_class_ids"]
    if len(target_ids) != logits.shape[0] or any(not ids for ids in target_ids):
        raise ValueError("one non-empty target class set is required per parent")
    log_normalizer = torch.logsumexp(logits, dim=1)
    losses = []
    for index, ids in enumerate(target_ids):
        selected = torch.as_tensor(ids, dtype=torch.long, device=logits.device)
        losses.append(log_normalizer[index] - torch.logsumexp(logits[index, selected], dim=0))
    return torch.stack(losses).mean()


def fused_probabilities(outputs):
    return F.softmax(outputs["fused"], dim=1)


def initialize_exact(model, checkpoint_path, checkpoint_sha256, source_class_map,
                     source_class_map_sha256, target_class_keys):
    """755 -> 1224 retains exact prefix class keys, with no name normalization.

    Bare legacy state_dicts require a separately bound source class table.
    Missing backbone tensors, extra keys, and incompatible dimensions fail.
    """
    import io
    source_data = json.loads(verified_bytes(source_class_map, source_class_map_sha256))
    classes = source_data["classes"]
    if ([c["model_class_id"] for c in classes] != list(range(len(classes)))
            or any(type(c["model_class_id"]) is not int for c in classes)):
        raise ValueError("invalid source class map order")
    source_keys = [c["class_key"] for c in classes]
    if (not source_keys or len(set(source_keys)) != len(source_keys)
            or target_class_keys[:len(source_keys)] != source_keys):
        raise ValueError("source class_key order is not the exact target prefix")
    payload = torch.load(io.BytesIO(verified_bytes(checkpoint_path, checkpoint_sha256)),
                         map_location="cpu", weights_only=False)
    if payload.get("checkpoint_format") in (LEGACY_FORMAT, CHECKPOINT_FORMAT):
        if payload["class_keys"] != source_keys:
            raise ValueError("checkpoint class_keys disagree with bound source class map")
        source = payload["model_state_dict"]
    else:
        source = payload
    target = model.state_dict()
    if set(source) != set(target):
        raise ValueError("checkpoint must contain exactly the shared backbone keys")
    count = len(source_keys)
    for key, tensor in source.items():
        expected = target[key]
        if key in (CLASSIFIER_WEIGHT_KEY, CLASSIFIER_BIAS_KEY):
            if tensor.shape != (count, *expected.shape[1:]):
                raise ValueError("source classifier shape disagrees with source class map")
        elif tensor.shape != expected.shape:
            raise ValueError(f"backbone shape mismatch: {key}")
    # Validate the entire checkpoint before modifying even one target tensor.
    with torch.no_grad():
        for key, tensor in source.items():
            if key in (CLASSIFIER_WEIGHT_KEY, CLASSIFIER_BIAS_KEY):
                target[key][:count].copy_(tensor)
            else:
                target[key].copy_(tensor)
    model.load_state_dict(target, strict=True)
    return count


def source_identity(source_dir=None):
    """Bind an exact Git checkout or a verified minimal source export.

    source_snapshot.json is generated by the exporter from committed files. It
    keeps the upstream revision available on hosts without a .git directory;
    a changed/missing source must fail rather than silently use an ancestor repo.
    """
    source_dir = Path(source_dir) if source_dir is not None else Path(__file__).resolve().parent
    sources = {}
    for name in ("dual_view_training_data.py", "dual_view_classifier_train.py",
                 "training_runtime.py"):
        sources[name] = digest((source_dir / name).read_bytes())
    snapshot_path = source_dir.parent / "source_snapshot.json"
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot["format"] != "birdsvision-dual-view-source-snapshot-v1":
            raise ValueError("invalid source snapshot format")
        commit = snapshot["git_commit"]
        if (not isinstance(commit, str) or len(commit) != 40
                or any(char not in "0123456789abcdef" for char in commit)):
            raise ValueError("invalid source snapshot git_commit")
        if snapshot["source_sha256"] != sources:
            raise ValueError("source snapshot SHA-256 mismatch")
    else:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_dir, text=True
        ).strip()
    return commit, sources


def runtime_contract(device):
    commit, sources = source_identity()
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "timm", "Pillow")}
    value = {"git_commit": commit, "source_sha256": sources,
             "python": platform.python_version(), "platform": platform.platform(),
             "packages": versions, "device": str(device), "cuda_runtime": torch.version.cuda,
             "cudnn_version": torch.backends.cudnn.version(),
             "torch_num_threads": torch.get_num_threads(),
             "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        value.update(gpu_name=props.name, gpu_total_memory=props.total_memory)
    return value


def build_run_contract(bundle, settings, data_config, transforms, environment):
    if settings["alpha"] != 0.5:
        raise ValueError("first baseline requires alpha=0.5")
    # Normalize tuples etc. once, then strict structural equality on resume.
    return json.loads(canonical({
        "format": CHECKPOINT_FORMAT, "data_contract_sha256": bundle.contract_sha256,
        "data_contract": bundle.contract, "class_keys": [c["class_key"] for c in bundle.classes],
        "crop_rule": CROP_RULE, "fusion": FUSION_RULE, "settings": settings,
        "preprocessing": {"data_config": data_config,
                          "full_train": repr(transforms[0]), "crop_train": repr(transforms[0]),
                          "full_validation": repr(transforms[1]), "crop_validation": repr(transforms[1])},
        "environment": environment,
        "metric_rules": {"area": "mean_final_unexpanded_bbox_area_per_parent",
                         "area_edges": [0.01, 0.1], "existing_class_ids": [0, 754],
                         "macro": "mean_top_accuracy_over_present_classes",
                         "unit": "parent"},
    }))


def validate_resume(checkpoint, contract):
    if checkpoint["checkpoint_format"] != CHECKPOINT_FORMAT:
        raise ValueError("not a complete dual-view checkpoint")
    if checkpoint["run_contract"] != contract or checkpoint["class_keys"] != contract["class_keys"]:
        raise ValueError("resume contract mismatch; data/fusion/crop/classes/runtime/code must match")
    if checkpoint["status"] not in ("epoch_complete", "interrupted_partial_epoch"):
        raise ValueError("unsupported checkpoint status")
    progress = checkpoint["progress"]
    if any(type(progress[k]) is not int or progress[k] < 0 for k in ("epoch", "next_batch", "no_improvement")):
        raise ValueError("invalid resume progress")


def save_checkpoint(path, model, optimizer, scheduler, scaler, contract, progress, status):
    payload = {"checkpoint_format": CHECKPOINT_FORMAT, "status": status,
               "class_keys": contract["class_keys"], "run_contract": copy.deepcopy(contract),
               "progress": copy.deepcopy(progress), "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
               # PyTorch 2.9.1's scheduler wrapper sets this outside state_dict.
               # Preserve it when resuming directly into validation/scheduler.
               "optimizer_step_called": getattr(optimizer, "_opt_called", False),
               "scaler_state_dict": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
               "python_rng": random.getstate(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.device(contract["environment"]["device"]).type == "cuda" else None}
    atomic_torch_save(payload, path)


def restore_checkpoint(checkpoint, contract, model, optimizer, scheduler, scaler):
    validate_resume(checkpoint, contract)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    optimizer._opt_called = checkpoint["optimizer_step_called"]
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng"].cpu())
    random.setstate(checkpoint["python_rng"])
    if checkpoint["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    return copy.deepcopy(checkpoint["progress"])


class ParentMetrics:
    def __init__(self, num_classes, state=None, *, class_to_species_group=None,
                 same_species_credit=0.85):
        self.num_classes = num_classes
        self.class_to_species_group = (
            list(class_to_species_group) if class_to_species_group is not None
            else [None] * num_classes
        )
        if len(self.class_to_species_group) != num_classes:
            raise ValueError("class_to_species_group length mismatch")
        if same_species_credit != 0.85:
            raise ValueError("hierarchical same-species credit must be 0.85")
        self.same_species_credit = same_species_credit
        self.state = copy.deepcopy(state) if state is not None else {
            "groups": {}, "loss_sum": 0.0, "parent_count": 0, "batch_view_counts": [],
            "elapsed_seconds": 0.0, "peak_cuda_memory_bytes": None}

    def update(self, outputs, batch, loss):
        labels = batch["labels"].tolist()
        predictions = {name: outputs[name].detach().topk(min(3, self.num_classes), dim=1).indices.cpu().tolist()
                       for name in ("full", "crop", "fused")}
        for index, (label, record) in enumerate(zip(labels, batch["records"], strict=True)):
            targets = batch["target_class_ids"][index]
            coarse = label < 0
            instances = record["instances"]
            area = sum((i["bbox_xyxy_normalized"][2] - i["bbox_xyxy_normalized"][0]) *
                       (i["bbox_xyxy_normalized"][3] - i["bbox_xyxy_normalized"][1]) for i in instances) / len(instances)
            area_group = "area_lt_0.01" if area < 0.01 else "area_0.01_to_0.1" if area < 0.1 else "area_ge_0.1"
            groups = ["overall", "single" if len(instances) == 1 else "multiple", area_group,
                      ("species_coarse" if coarse else
                       "existing_0_754" if label < 755 else "new_755_plus"),
                      ("species:" + record["supervision"]["species_catalog_entry_id"]
                       if coarse else f"class:{label}")]
            for group in groups:
                entry = self.state["groups"].setdefault(group, {"parents": 0, **{
                    f"{name}_{metric}": 0 for name in predictions
                    for metric in ("top1", "top3", "hierarchical_top1")}})
                entry["parents"] += 1
                for name, predicted in predictions.items():
                    top = predicted[index]
                    entry[f"{name}_top1"] += int(top[0] in targets)
                    entry[f"{name}_top3"] += int(any(value in targets for value in top))
                    if coarse or top[0] == label:
                        credit = float(top[0] in targets)
                    else:
                        true_group = self.class_to_species_group[label]
                        predicted_group = self.class_to_species_group[top[0]]
                        credit = self.same_species_credit if (
                            true_group is not None and predicted_group == true_group
                        ) else 0.0
                    entry[f"{name}_hierarchical_top1"] += credit
        self.state["loss_sum"] += float(loss) * len(labels)
        self.state["parent_count"] += len(labels)
        self.state["batch_view_counts"].append(len(batch["flat_views"]))

    def report(self):
        state = self.state
        count = state["parent_count"]
        if not count:
            raise ValueError("no parent metrics")
        groups = {name: {"parents": values["parents"], **{
            key: 100.0 * number / values["parents"] for key, number in values.items() if key != "parents"}}
            for name, values in state["groups"].items()}
        class_groups = [v for k, v in groups.items()
                        if k.startswith("class:") or k.startswith("species:")]
        macro = {key: sum(v[key] for v in class_groups) / len(class_groups)
                 for key in class_groups[0] if key != "parents"}
        return {"parent_count": count, "loss": state["loss_sum"] / count, "groups": groups,
                "macro_present_classes": macro,
                "class_sample_counts": [state["groups"].get(f"class:{i}", {}).get("parents", 0)
                                        for i in range(self.num_classes)],
                "batch_view_counts": state["batch_view_counts"],
                "elapsed_seconds": state["elapsed_seconds"],
                "parents_per_second": count / max(state["elapsed_seconds"], 1e-9)}


def make_loader(bundle, split, transform, settings, epoch):
    dataset = DualViewDataset(bundle, split, transform, seed=settings["seed"], epoch=epoch)
    sampler = ViewBudgetBatchSampler(dataset.records, settings["max_views"],
                                    shuffle=split == "train", seed=settings["seed"] + epoch)
    generator = torch.Generator().manual_seed(settings["seed"] + epoch)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=settings["num_workers"],
                      collate_fn=partial(collate_views, max_views=settings["max_views"]), generator=generator)


def run_epochs(bundle, model, transforms, contract, output_dir, *, resume_path=None,
               stop_request=None, max_runtime_minutes=0):
    """Engine shared with fake-based tests. Public training authorization is in main."""
    settings = contract["settings"]
    device = torch.device(contract["environment"]["device"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])
    amp = settings["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    progress = {"epoch": 0, "next_batch": 0, "best_accuracy": -1.0, "no_improvement": 0, "train_metrics": None}
    if resume_path is not None:
        # Validate before any writes, including run_contract.json.
        progress = restore_checkpoint(load_torch_file(resume_path, device), contract, model, optimizer, scheduler, scaler)
    if output_dir.exists() and resume_path is None:
        raise FileExistsError("new runs require a new output directory")
    if resume_path is not None:
        existing = json.loads((output_dir / "run_contract.json").read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError("output run contract mismatch")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        atomic_json_save(contract, output_dir / "run_contract.json")
    stop = stop_request or StopRequest()
    previous = {sig: signal.signal(sig, stop.request) for sig in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    def interrupted():
        save_checkpoint(output_dir / "interrupted.checkpoint.pth", model, optimizer, scheduler, scaler,
                        contract, progress, "interrupted_partial_epoch")
        return 130 if stop.signal_number == signal.SIGINT else 143
    try:
        # A terminal early-stop checkpoint remains terminal on resume.
        if settings["patience"] and progress["no_improvement"] >= settings["patience"]:
            return 0
        for epoch in range(progress["epoch"], settings["epochs"]):
            if progress["next_batch"] == 0:
                seed_everything(settings["seed"] + epoch)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_loader = make_loader(bundle, "train", transforms[0], settings, epoch)
            if progress["next_batch"] > len(train_loader):
                raise ValueError("resume batch cursor exceeds the logical epoch")
            metrics = ParentMetrics(
                len(bundle.classes), progress["train_metrics"],
                class_to_species_group=bundle.contract.get("class_hierarchy", {}).get(
                    "class_to_species_group"
                ),
            )
            model.train()
            tick = time.monotonic()
            for index, batch in enumerate(train_loader):
                if index < progress["next_batch"]:
                    tick = time.monotonic()
                    continue
                if stop.signal_number is not None:
                    return interrupted()
                optimizer.zero_grad(set_to_none=True)
                outputs = forward_batch(model, batch, device, alpha=settings["alpha"], amp=amp)
                loss = parent_loss(outputs, batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                metrics.update(outputs, batch, loss.detach().item())
                metrics.state["elapsed_seconds"] += time.monotonic() - tick
                if device.type == "cuda":
                    metrics.state["peak_cuda_memory_bytes"] = max(
                        metrics.state["peak_cuda_memory_bytes"] or 0, torch.cuda.max_memory_allocated(device))
                progress.update(epoch=epoch, next_batch=index + 1, train_metrics=metrics.state)
                if stop.signal_number is not None:
                    return interrupted()
                tick = time.monotonic()
            validation = ParentMetrics(
                len(bundle.classes),
                class_to_species_group=bundle.contract.get("class_hierarchy", {}).get(
                    "class_to_species_group"
                ),
            )
            model.eval()
            tick = time.monotonic()
            with torch.no_grad():
                for batch in make_loader(bundle, "validation", transforms[1], settings, epoch):
                    if stop.signal_number is not None:
                        return interrupted()
                    outputs = forward_batch(model, batch, device, alpha=settings["alpha"], amp=amp)
                    loss = parent_loss(outputs, batch)
                    validation.update(outputs, batch, loss.item())
            if stop.signal_number is not None:
                return interrupted()
            validation.state["elapsed_seconds"] = time.monotonic() - tick
            validation_report = validation.report()
            accuracy = validation_report["groups"]["overall"]["fused_hierarchical_top1"]
            improved = accuracy > progress["best_accuracy"]
            progress["best_accuracy"] = max(accuracy, progress["best_accuracy"])
            progress["no_improvement"] = 0 if improved else progress["no_improvement"] + 1
            progress.update(epoch=epoch + 1, next_batch=0, train_metrics=None)
            scheduler.step()
            save_checkpoint(output_dir / "latest.checkpoint.pth", model, optimizer, scheduler, scaler,
                            contract, progress, "epoch_complete")
            if improved:
                save_checkpoint(output_dir / "best.checkpoint.pth", model, optimizer, scheduler, scaler,
                                contract, progress, "epoch_complete")
            result = {"completed_epochs": epoch + 1, "train": metrics.report(), "validation": validation_report,
                      "zero_box_exclusions": bundle.contract["parent_summary"],
                      "peak_cuda_memory_bytes": max(metrics.state["peak_cuda_memory_bytes"] or 0,
                                                    torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None}
            atomic_json_save(result, output_dir / f"epoch-{epoch + 1:04d}.json")
            print(json.dumps({"epoch": epoch + 1, "validation_fused_top1": accuracy,
                              "train_parents": metrics.state["parent_count"]}), flush=True)
            if settings["patience"] and progress["no_improvement"] >= settings["patience"]:
                return 0
            if max_runtime_minutes and time.monotonic() - started >= max_runtime_minutes * 60:
                return 0
        return 0
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def synthetic_smoke():
    seed_everything(20260907)
    with tempfile.TemporaryDirectory(prefix="birdsvision-dual-view-") as temporary:
        bundle = synthetic_bundle(Path(temporary) / "data")
        model = StableCUBModel(num_classes=1224, use_pretrained=False).cpu().eval()
        _, transform, _ = make_transforms(model)
        dataset = DualViewDataset(bundle, "train", transform)
        batch = collate_views([dataset[0], dataset[1]], max_views=5)
        calls = []
        handle = model.register_forward_hook(lambda _m, args, _out: calls.append(list(args[0].shape)))
        tick = time.monotonic()
        try:
            with torch.no_grad():
                outputs = forward_batch(model, batch, torch.device("cpu"))
                probabilities = fused_probabilities(outputs)
        finally:
            handle.remove()
        for record in bundle.records + bundle.exclusions:
            verified_bytes(relative_file(bundle.root, record["storage_path"]), record["sha256"], record["byte_size"])
        assert calls == [[5, 3, 224, 224]]
        assert list(outputs["fused"].shape) == [2, 1224]
        assert torch.allclose(probabilities.sum(1), torch.ones(2))
        assert all(key.startswith("backbone.") for key in model.state_dict())
        return {"status": "synthetic_cpu_forward_only", "forward_calls": len(calls),
                "input_shape": calls[0], "fused_shape": list(outputs["fused"].shape),
                "crop_counts": outputs["crop_counts"].tolist(), "source_hashes_unchanged": True,
                "elapsed_seconds": time.monotonic() - tick,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "parameter_bytes_fp32": sum(p.numel() * p.element_size() for p in model.parameters()),
                "environment": runtime_contract(torch.device("cpu"))}


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if len(raw_argv) == 2 and raw_argv[0] == CONFIG_ARGUMENT:
        return load_training_config(Path(raw_argv[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(CONFIG_ARGUMENT, type=Path,
                        help=f"config-only mode; reads {CONFIG_ATTRIBUTE} and accepts no other options")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--data-contract", type=Path)
    parser.add_argument("--data-contract-sha256")
    parser.add_argument("--allow-training", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-views", type=positive_int)
    parser.add_argument("--epochs", type=positive_int, default=25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--alpha", type=float, choices=(0.5,), default=0.5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--max-runtime-minutes", type=float, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--init-checkpoint-sha256")
    parser.add_argument("--init-class-map", type=Path)
    parser.add_argument("--init-class-map-sha256")
    args = parser.parse_args(raw_argv)
    if args.config is not None:
        parser.error(f"{CONFIG_ARGUMENT} must be the only option")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.synthetic_smoke:
            if (args.allow_training or args.data_contract or args.resume or args.init_checkpoint
                    or args.device != "cpu" or args.output_dir or args.project_root):
                raise ValueError("synthetic smoke is isolated CPU forward only")
            print(json.dumps(synthetic_smoke(), indent=2))
            return 0
        if not all((args.project_root, args.data_contract, args.data_contract_sha256, args.max_views)):
            raise ValueError("explicit project-root, data-contract, data-contract-sha256 and max-views required")
        for name in ("learning_rate", "weight_decay", "max_runtime_minutes"):
            value = getattr(args, name)
            if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
                raise ValueError(f"invalid {name}")
        if args.num_workers < 0 or args.patience < 0:
            raise ValueError("num-workers and patience must be nonnegative")
        bundle = load_bundle(args.project_root, args.data_contract, args.data_contract_sha256)
        for split in ("train", "validation"):
            ViewBudgetBatchSampler([r for r in bundle.records if r["split"] == split], args.max_views)
        if not args.allow_training:
            print(json.dumps({"status": "preflight_only", "data_contract_sha256": bundle.contract_sha256,
                              "parent_summary": bundle.contract["parent_summary"]}, indent=2))
            return 0
        if args.output_dir is None:
            raise ValueError("training requires an explicit new output-dir")
        if args.resume == "auto":
            args.resume = resolve_auto_resume(args.output_dir)
        initialization_args = (args.init_checkpoint, args.init_checkpoint_sha256, args.init_class_map, args.init_class_map_sha256)
        if args.resume is None and not all(initialization_args):
            raise ValueError("training requires exact initialization checkpoint and source class map hashes")
        if args.resume is not None and any(initialization_args):
            raise ValueError("resume and initialization are mutually exclusive")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable")
        seed_everything(args.seed)
        model = StableCUBModel(num_classes=len(bundle.classes), use_pretrained=False)
        train_transform, validation_transform, data_config = make_transforms(model)
        settings = {key: getattr(args, key) for key in (
            "max_views", "epochs", "num_workers", "learning_rate", "weight_decay", "seed", "alpha", "amp", "patience")}
        if args.resume:
            previous = load_torch_file(args.resume, torch.device("cpu"))
            settings["initialization"] = previous["run_contract"]["settings"]["initialization"]
        else:
            settings["initialization"] = {"checkpoint_sha256": args.init_checkpoint_sha256,
                                          "source_class_map_sha256": args.init_class_map_sha256}
            initialize_exact(model, *initialization_args, [c["class_key"] for c in bundle.classes])
        contract = build_run_contract(bundle, settings, data_config, (train_transform, validation_transform), runtime_contract(device))
        return run_epochs(bundle, model, (train_transform, validation_transform), contract,
                          args.output_dir.resolve(), resume_path=args.resume, max_runtime_minutes=args.max_runtime_minutes)
    except Exception as exc:
        print(f"[dual-view] failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
