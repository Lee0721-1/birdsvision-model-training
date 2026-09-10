#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Independent low-rate dual-view fine-tuning with adaptive logits fusion.

The only formal entry is ``--config``.  The data loader accepts train and
validation only; there is no final-test argument or manifest route.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import io
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
import warnings
from functools import partial
from pathlib import Path


warnings.filterwarnings(
    "ignore",
    message=r"^Truncated File Read$",
    category=UserWarning,
    module=r"^PIL\.TiffImagePlugin$",
)

import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training_runtime import (
    StableCUBModel,
    StopRequest,
    atomic_json_save,
    atomic_torch_save,
    load_torch_file,
    make_transforms,
    seed_everything,
)
from dual_view_classifier_train import (
    CHECKPOINT_FORMAT as BASE_CHECKPOINT_FORMAT,
    ParentMetrics,
    fuse_logits,
)
from dual_view_finetune_training_data import FineTuneDualViewDataset
from dual_view_training_data import (
    CROP_RULE,
    ZERO_RULE,
    ViewBudgetBatchSampler,
    canonical,
    collate_views,
    digest,
    load_bundle,
    synthetic_bundle,
    verified_bytes,
)


CHECKPOINT_FORMAT = "birdsvision-dual-view-finetune-checkpoint-v1"
CONFIG_ATTRIBUTE = "DUAL_VIEW_FINE_TUNING_RUN"
CONFIG_ARGUMENT = "--config"
CONFIG_KEYS = frozenset({
    "allow_training", "project_root", "data_contract", "data_contract_sha256",
    "output_dir", "device", "max_views", "epochs", "num_workers",
    "learning_rate", "weight_decay", "seed", "default_alpha",
    "many_box_threshold", "many_box_alpha", "full_aux_loss_weight",
    "crop_aux_loss_weight", "tiny_crop_area_threshold", "tiny_crop_scale_min",
    "amp", "patience", "max_runtime_minutes", "resume", "init_checkpoint",
    "init_checkpoint_sha256",
})
CONFIG_PATH_KEYS = frozenset({
    "project_root", "data_contract", "output_dir", "init_checkpoint",
})


def load_training_config(config_path):
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"training config does not exist: {config_path}")
    raw = runpy.run_path(str(config_path)).get(CONFIG_ATTRIBUTE)
    if not isinstance(raw, dict):
        raise ValueError(f"{CONFIG_ATTRIBUTE} must be a dictionary")
    missing, unknown = CONFIG_KEYS - set(raw), set(raw) - CONFIG_KEYS
    if missing or unknown:
        raise ValueError(
            f"invalid {CONFIG_ATTRIBUTE} keys; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    values = dict(raw)
    for key in CONFIG_PATH_KEYS:
        value = values[key]
        if value is None:
            if key not in ("output_dir", "init_checkpoint"):
                raise ValueError(f"{key} cannot be None")
            continue
        if not isinstance(value, (str, Path)):
            raise ValueError(f"{key} must be a path")
        path = Path(value).expanduser()
        values[key] = ((config_path.parent / path).resolve()
                       if not path.is_absolute() else path.resolve())
    resume = values["resume"]
    if resume != "auto" and resume is not None:
        if not isinstance(resume, (str, Path)):
            raise ValueError("resume must be None, 'auto', or a checkpoint path")
        path = Path(resume).expanduser()
        values["resume"] = ((config_path.parent / path).resolve()
                            if not path.is_absolute() else path.resolve())
    for key in ("allow_training", "amp"):
        if type(values[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    return argparse.Namespace(synthetic_smoke=False, config=config_path, **values)


def resolve_auto_resume(output_dir):
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
            if any(type(progress.get(key)) is not int
                   for key in ("epoch", "next_batch")):
                raise ValueError(f"auto-resume found invalid progress: {path}")
            choices.append(((progress["epoch"], progress["next_batch"]), path))
    if not choices:
        raise FileExistsError("configured output_dir exists without a resumable checkpoint")
    return max(choices, key=lambda value: value[0])[1]


def adaptive_fuse_logits(flat_logits, view_parent, view_role, parent_count, *,
                         default_alpha, many_box_threshold, many_box_alpha):
    if type(many_box_threshold) is not int or many_box_threshold < 1:
        raise ValueError("many_box_threshold must be a positive integer")
    for name, value in (("default_alpha", default_alpha),
                        ("many_box_alpha", many_box_alpha)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite in [0, 1]")
    outputs = fuse_logits(
        flat_logits, view_parent, view_role, parent_count, alpha=0.0
    )
    alpha = torch.full(
        (parent_count, 1), float(default_alpha),
        dtype=outputs["full"].dtype, device=outputs["full"].device,
    )
    alpha = torch.where(
        (outputs["crop_counts"] > many_box_threshold).unsqueeze(1),
        alpha.new_full(alpha.shape, float(many_box_alpha)),
        alpha,
    )
    outputs["fused"] = torch.where(
        (outputs["crop_counts"] > 0).unsqueeze(1),
        alpha * outputs["full"] + (1 - alpha) * outputs["crop"],
        outputs["full"],
    )
    outputs["alpha"] = alpha.squeeze(1)
    return outputs


def forward_batch(classifier, batch, device, settings, *, amp=False):
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        flat_logits = classifier(batch["flat_views"].to(device))
    return adaptive_fuse_logits(
        flat_logits, batch["view_parent"], batch["view_role"],
        len(batch["labels"]),
        default_alpha=settings["default_alpha"],
        many_box_threshold=settings["many_box_threshold"],
        many_box_alpha=settings["many_box_alpha"],
    )


def supervision_loss(logits, batch):
    target_ids = batch["target_class_ids"]
    if len(target_ids) != logits.shape[0] or any(not ids for ids in target_ids):
        raise ValueError("one non-empty target class set is required per parent")
    normalizer = torch.logsumexp(logits, dim=1)
    losses = []
    for index, ids in enumerate(target_ids):
        selected = torch.as_tensor(ids, dtype=torch.long, device=logits.device)
        losses.append(normalizer[index] - torch.logsumexp(logits[index, selected], dim=0))
    return torch.stack(losses).mean()


def parent_losses(outputs, batch, *, full_weight, crop_weight):
    if torch.any(outputs["crop_counts"] == 0):
        raise ValueError("zero-box parents cannot receive supervised loss")
    for name, value in (("full_weight", full_weight), ("crop_weight", crop_weight)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    fused = supervision_loss(outputs["fused"], batch)
    full = supervision_loss(outputs["full"], batch)
    crop = supervision_loss(outputs["crop"], batch)
    total = (fused + full_weight * full + crop_weight * crop) / (
        1 + full_weight + crop_weight
    )
    return {"total": total, "fused": fused, "full": full, "crop": crop}


def load_initial_checkpoint(model, path, expected_sha256, class_keys):
    payload = torch.load(
        io.BytesIO(verified_bytes(path, expected_sha256)),
        map_location="cpu", weights_only=False,
    )
    if payload.get("checkpoint_format") != BASE_CHECKPOINT_FORMAT:
        raise ValueError("initial checkpoint is not a completed dual-view baseline")
    if payload.get("status") != "epoch_complete":
        raise ValueError("initial checkpoint must end at a completed epoch")
    if payload.get("class_keys") != class_keys:
        raise ValueError("initial checkpoint class table mismatch")
    progress = payload.get("progress", {})
    if type(progress.get("epoch")) is not int or progress["epoch"] <= 0:
        raise ValueError("initial checkpoint progress is invalid")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return {
        "checkpoint_format": payload["checkpoint_format"],
        "checkpoint_sha256": expected_sha256,
        "completed_epoch": progress["epoch"],
        "source_run_contract_sha256": digest(canonical(payload["run_contract"])),
    }


def source_identity(source_dir=None):
    source_dir = (Path(source_dir) if source_dir is not None
                  else Path(__file__).resolve().parent)
    names = (
        "dual_view_training_data.py", "dual_view_classifier_train.py",
        "dual_view_finetune_training_data.py", "dual_view_classifier_finetune.py",
        "training_runtime.py",
    )
    sources = {name: digest((source_dir / name).read_bytes()) for name in names}
    snapshot_path = source_dir.parent / "source_snapshot.json"
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot["format"] != "birdsvision-dual-view-finetune-source-snapshot-v1":
            raise ValueError("invalid fine-tune source snapshot format")
        commit = snapshot["git_commit"]
        if snapshot["source_sha256"] != sources:
            raise ValueError("source snapshot SHA-256 mismatch")
    else:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_dir, text=True
        ).strip()
    if (not isinstance(commit, str) or len(commit) != 40
            or any(char not in "0123456789abcdef" for char in commit)):
        raise ValueError("invalid source git commit")
    return commit, sources


def runtime_contract(device):
    commit, sources = source_identity()
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "timm", "Pillow")}
    value = {
        "git_commit": commit, "source_sha256": sources,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": versions, "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        value.update(gpu_name=properties.name,
                     gpu_total_memory=properties.total_memory)
    return value


def build_run_contract(bundle, settings, data_config, transforms, environment):
    train_transform, validation_transform, tiny_transform = transforms
    return json.loads(canonical({
        "format": CHECKPOINT_FORMAT,
        "data_contract_sha256": bundle.contract_sha256,
        "data_contract": bundle.contract,
        "class_keys": [item["class_key"] for item in bundle.classes],
        "crop_rule": CROP_RULE,
        "fusion": {
            "location": "logits",
            "default_alpha": settings["default_alpha"],
            "many_box_condition": f"crop_count > {settings['many_box_threshold']}",
            "many_box_alpha": settings["many_box_alpha"],
            "multi_box_aggregation": "equal_mean_crop_logits_1_over_N",
            "normal_inference_zero_box_rule": "exact_full_only",
            "softmax": "after_fusion_only",
        },
        "loss": {
            "unit": "mean_over_parents",
            "fused_weight": 1.0,
            "full_auxiliary_weight": settings["full_aux_loss_weight"],
            "crop_auxiliary_weight": settings["crop_aux_loss_weight"],
            "normalization": "divide_by_sum_of_component_weights",
            "supervised_zero_box_rule": ZERO_RULE,
            "supervision": "exact_or_species_marginal_nll",
        },
        "settings": settings,
        "preprocessing": {
            "data_config": data_config,
            "full_train": repr(train_transform),
            "crop_train": repr(train_transform),
            "tiny_crop_train": repr(tiny_transform),
            "tiny_crop_condition": (
                "unexpanded_bbox_area < "
                f"{settings['tiny_crop_area_threshold']}"
            ),
            "full_validation": repr(validation_transform),
            "crop_validation": repr(validation_transform),
        },
        "environment": environment,
        "metric_rules": {
            "area": "mean_final_unexpanded_bbox_area_per_parent",
            "area_edges": [0.01, 0.1],
            "existing_class_ids": [0, 754],
            "macro": "mean_top_accuracy_over_present_classes",
            "unit": "parent",
        },
    }))


def save_checkpoint(path, model, optimizer, scheduler, scaler, contract,
                    progress, status):
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "status": status,
        "class_keys": contract["class_keys"],
        "run_contract": copy.deepcopy(contract),
        "progress": copy.deepcopy(progress),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "optimizer_step_called": getattr(optimizer, "_opt_called", False),
        "scaler_state_dict": scaler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "cuda_rng": (torch.cuda.get_rng_state_all()
                     if torch.device(contract["environment"]["device"]).type == "cuda"
                     else None),
    }
    atomic_torch_save(payload, path)


def restore_checkpoint(checkpoint, contract, model, optimizer, scheduler, scaler):
    if checkpoint.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError("not a complete fine-tune checkpoint")
    if (checkpoint.get("run_contract") != contract
            or checkpoint.get("class_keys") != contract["class_keys"]):
        raise ValueError("resume contract mismatch")
    if checkpoint.get("status") not in ("epoch_complete", "interrupted_partial_epoch"):
        raise ValueError("unsupported checkpoint status")
    progress = checkpoint["progress"]
    if any(type(progress.get(key)) is not int or progress[key] < 0
           for key in ("epoch", "next_batch", "no_improvement")):
        raise ValueError("invalid resume progress")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    optimizer._opt_called = checkpoint["optimizer_step_called"]
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng"].cpu())
    random.setstate(checkpoint["python_rng"])
    if checkpoint["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    return copy.deepcopy(progress)


def make_loader(bundle, split, transforms, settings, epoch):
    train_transform, validation_transform, tiny_transform = transforms
    if split == "train":
        full_transform = crop_transform = train_transform
        tiny = tiny_transform
    else:
        full_transform = crop_transform = validation_transform
        tiny = None
    dataset = FineTuneDualViewDataset(
        bundle, split, full_transform, crop_transform, tiny,
        tiny_crop_area_threshold=settings["tiny_crop_area_threshold"],
        seed=settings["seed"], epoch=epoch,
    )
    sampler = ViewBudgetBatchSampler(
        dataset.records, settings["max_views"], shuffle=split == "train",
        seed=settings["seed"] + epoch,
    )
    generator = torch.Generator().manual_seed(settings["seed"] + epoch)
    return DataLoader(
        dataset, batch_sampler=sampler, num_workers=settings["num_workers"],
        collate_fn=partial(collate_views, max_views=settings["max_views"]),
        generator=generator,
    )


def update_loss_components(metrics, losses, parent_count):
    sums = metrics.state.setdefault(
        "component_loss_sums", {name: 0.0 for name in ("total", "fused", "full", "crop")}
    )
    for name in sums:
        sums[name] += float(losses[name].detach()) * parent_count


def metrics_report(metrics):
    report = metrics.report()
    sums = metrics.state.get("component_loss_sums")
    if sums is not None:
        report["loss_components"] = {
            name: value / metrics.state["parent_count"] for name, value in sums.items()
        }
    return report


def run_epochs(bundle, model, transforms, contract, output_dir, *,
               resume_path=None, stop_request=None, max_runtime_minutes=0):
    settings = contract["settings"]
    device = torch.device(contract["environment"]["device"])
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["epochs"]
    )
    amp = settings["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    progress = {
        "epoch": 0, "next_batch": 0, "best_accuracy": -1.0,
        "no_improvement": 0, "train_metrics": None,
    }
    if resume_path is not None:
        progress = restore_checkpoint(
            load_torch_file(resume_path, device), contract, model,
            optimizer, scheduler, scaler,
        )
    output_dir = Path(output_dir)
    if output_dir.exists() and resume_path is None:
        raise FileExistsError("new runs require a new output directory")
    if resume_path is None:
        output_dir.mkdir(parents=True, exist_ok=False)
        atomic_json_save(contract, output_dir / "run_contract.json")
    elif json.loads((output_dir / "run_contract.json").read_text(encoding="utf-8")) != contract:
        raise ValueError("output run contract mismatch")
    stop = stop_request or StopRequest()
    previous = {sig: signal.signal(sig, stop.request)
                for sig in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()

    def interrupted():
        save_checkpoint(
            output_dir / "interrupted.checkpoint.pth", model, optimizer,
            scheduler, scaler, contract, progress, "interrupted_partial_epoch",
        )
        return 130 if stop.signal_number == signal.SIGINT else 143

    try:
        if settings["patience"] and progress["no_improvement"] >= settings["patience"]:
            return 0
        for epoch in range(progress["epoch"], settings["epochs"]):
            if progress["next_batch"] == 0:
                seed_everything(settings["seed"] + epoch)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_loader = make_loader(bundle, "train", transforms, settings, epoch)
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
                outputs = forward_batch(model, batch, device, settings, amp=amp)
                losses = parent_losses(
                    outputs, batch,
                    full_weight=settings["full_aux_loss_weight"],
                    crop_weight=settings["crop_aux_loss_weight"],
                )
                scaler.scale(losses["total"]).backward()
                scaler.step(optimizer)
                scaler.update()
                parent_count = len(batch["labels"])
                metrics.update(outputs, batch, losses["total"].detach().item())
                update_loss_components(metrics, losses, parent_count)
                metrics.state["elapsed_seconds"] += time.monotonic() - tick
                if device.type == "cuda":
                    metrics.state["peak_cuda_memory_bytes"] = max(
                        metrics.state["peak_cuda_memory_bytes"] or 0,
                        torch.cuda.max_memory_allocated(device),
                    )
                progress.update(
                    epoch=epoch, next_batch=index + 1, train_metrics=metrics.state
                )
                if ((index + 1) % 20 == 0 or index + 1 == len(train_loader)):
                    print(json.dumps({
                        "phase": "train", "epoch": epoch + 1,
                        "batch": index + 1, "batches": len(train_loader),
                        "loss": losses["total"].detach().item(),
                        "views": len(batch["flat_views"]),
                    }), flush=True)
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
                for batch in make_loader(bundle, "validation", transforms, settings, epoch):
                    if stop.signal_number is not None:
                        return interrupted()
                    outputs = forward_batch(model, batch, device, settings, amp=amp)
                    losses = parent_losses(
                        outputs, batch,
                        full_weight=settings["full_aux_loss_weight"],
                        crop_weight=settings["crop_aux_loss_weight"],
                    )
                    validation.update(outputs, batch, losses["total"].item())
                    update_loss_components(validation, losses, len(batch["labels"]))
            if stop.signal_number is not None:
                return interrupted()
            validation.state["elapsed_seconds"] = time.monotonic() - tick
            validation_report = metrics_report(validation)
            accuracy = validation_report["groups"]["overall"]["fused_hierarchical_top1"]
            improved = accuracy > progress["best_accuracy"]
            progress["best_accuracy"] = max(accuracy, progress["best_accuracy"])
            progress["no_improvement"] = 0 if improved else progress["no_improvement"] + 1
            progress.update(epoch=epoch + 1, next_batch=0, train_metrics=None)
            scheduler.step()
            save_checkpoint(
                output_dir / "latest.checkpoint.pth", model, optimizer, scheduler,
                scaler, contract, progress, "epoch_complete",
            )
            if improved:
                save_checkpoint(
                    output_dir / "best.checkpoint.pth", model, optimizer,
                    scheduler, scaler, contract, progress, "epoch_complete",
                )
            result = {
                "completed_epochs": epoch + 1,
                "train": metrics_report(metrics),
                "validation": validation_report,
                "zero_box_exclusions": bundle.contract["parent_summary"],
                "peak_cuda_memory_bytes": (
                    max(metrics.state["peak_cuda_memory_bytes"] or 0,
                        torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda" else None
                ),
            }
            atomic_json_save(result, output_dir / f"epoch-{epoch + 1:04d}.json")
            print(json.dumps({
                "epoch": epoch + 1,
                "validation_fused_hierarchical_top1": accuracy,
                "train_parents": metrics.state["parent_count"],
            }), flush=True)
            if settings["patience"] and progress["no_improvement"] >= settings["patience"]:
                return 0
            if max_runtime_minutes and time.monotonic() - started >= max_runtime_minutes * 60:
                return 0
        return 0
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def synthetic_smoke():
    seed_everything(20260908)
    with tempfile.TemporaryDirectory(prefix="birdsvision-dual-view-finetune-") as temporary:
        bundle = synthetic_bundle(Path(temporary) / "data")
        model = StableCUBModel(num_classes=1224, use_pretrained=False).cpu().eval()
        _, validation_transform, _ = make_transforms(model)
        dataset = FineTuneDualViewDataset(
            bundle, "train", validation_transform, validation_transform,
            validation_transform, tiny_crop_area_threshold=0.01,
        )
        batch = collate_views([dataset[0], dataset[1]], max_views=5)
        calls = []
        handle = model.register_forward_hook(
            lambda _model, args, _output: calls.append(list(args[0].shape))
        )
        try:
            with torch.no_grad():
                outputs = forward_batch(model, batch, torch.device("cpu"), {
                    "default_alpha": 0.4,
                    "many_box_threshold": 3,
                    "many_box_alpha": 0.25,
                })
                probabilities = F.softmax(outputs["fused"], dim=1)
        finally:
            handle.remove()
        assert calls == [[5, 3, 224, 224]]
        assert torch.allclose(probabilities.sum(1), torch.ones(2))
        return {
            "status": "synthetic_cpu_forward_only",
            "forward_calls": len(calls), "input_shape": calls[0],
            "fused_shape": list(outputs["fused"].shape),
            "crop_counts": outputs["crop_counts"].tolist(),
            "alpha": outputs["alpha"].tolist(),
        }


def parse_args(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if len(raw_argv) == 2 and raw_argv[0] == CONFIG_ARGUMENT:
        return load_training_config(Path(raw_argv[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(CONFIG_ARGUMENT, type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args(raw_argv)
    if args.config is not None:
        parser.error(f"{CONFIG_ARGUMENT} must be the only option")
    return args


def validate_settings(args):
    for name in (
        "learning_rate", "weight_decay", "max_runtime_minutes",
        "full_aux_loss_weight", "crop_aux_loss_weight",
    ):
        value = getattr(args, name)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (name == "learning_rate" and value == 0)):
            raise ValueError(f"invalid {name}")
    for name in ("epochs", "max_views"):
        if type(getattr(args, name)) is not int or getattr(args, name) <= 0:
            raise ValueError(f"invalid {name}")
    if (type(args.num_workers) is not int or args.num_workers < 0
            or type(args.patience) is not int or args.patience < 0):
        raise ValueError("num_workers and patience must be nonnegative integers")
    if type(args.many_box_threshold) is not int or args.many_box_threshold < 1:
        raise ValueError("many_box_threshold must be a positive integer")
    for name in ("default_alpha", "many_box_alpha"):
        value = getattr(args, name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid {name}")
    if (type(args.tiny_crop_area_threshold) not in (int, float)
            or not 0 < args.tiny_crop_area_threshold < 1):
        raise ValueError("invalid tiny_crop_area_threshold")
    if (type(args.tiny_crop_scale_min) not in (int, float)
            or not 0 < args.tiny_crop_scale_min <= 1):
        raise ValueError("invalid tiny_crop_scale_min")


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.synthetic_smoke:
            print(json.dumps(synthetic_smoke(), indent=2))
            return 0
        validate_settings(args)
        bundle = load_bundle(
            args.project_root, args.data_contract, args.data_contract_sha256
        )
        for split in ("train", "validation"):
            ViewBudgetBatchSampler(
                [record for record in bundle.records if record["split"] == split],
                args.max_views,
            )
        if not args.allow_training:
            print(json.dumps({
                "status": "preflight_only",
                "data_contract_sha256": bundle.contract_sha256,
                "parent_summary": bundle.contract["parent_summary"],
            }, indent=2))
            return 0
        if args.output_dir is None:
            raise ValueError("training requires an explicit output_dir")
        automatic_resume = args.resume == "auto"
        if automatic_resume:
            args.resume = resolve_auto_resume(args.output_dir)
        if args.resume is None and not all((args.init_checkpoint, args.init_checkpoint_sha256)):
            raise ValueError("new fine-tuning requires a hash-bound initial checkpoint")
        if (args.resume is not None and not automatic_resume
                and any((args.init_checkpoint, args.init_checkpoint_sha256))):
            raise ValueError("resume and initialization are mutually exclusive")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable")
        seed_everything(args.seed)
        model = StableCUBModel(num_classes=len(bundle.classes), use_pretrained=False)
        train_transform, validation_transform, data_config = make_transforms(model)
        tiny_transform = timm.data.create_transform(
            **data_config, is_training=True,
            scale=(float(args.tiny_crop_scale_min), 1.0),
        )
        settings = {key: getattr(args, key) for key in (
            "max_views", "epochs", "num_workers", "learning_rate",
            "weight_decay", "seed", "default_alpha", "many_box_threshold",
            "many_box_alpha", "full_aux_loss_weight", "crop_aux_loss_weight",
            "tiny_crop_area_threshold", "tiny_crop_scale_min", "amp", "patience",
        )}
        if args.resume is not None:
            previous = load_torch_file(args.resume, torch.device("cpu"))
            settings["initialization"] = previous["run_contract"]["settings"]["initialization"]
            if automatic_resume:
                configured_sha = args.init_checkpoint_sha256
                if configured_sha != settings["initialization"]["checkpoint_sha256"]:
                    raise ValueError("auto-resume initialization hash disagrees with run contract")
        else:
            settings["initialization"] = load_initial_checkpoint(
                model, args.init_checkpoint, args.init_checkpoint_sha256,
                [item["class_key"] for item in bundle.classes],
            )
        transforms = (train_transform, validation_transform, tiny_transform)
        contract = build_run_contract(
            bundle, settings, data_config, transforms, runtime_contract(device)
        )
        return run_epochs(
            bundle, model, transforms, contract, args.output_dir,
            resume_path=args.resume,
            max_runtime_minutes=args.max_runtime_minutes,
        )
    except Exception as exc:
        print(f"[dual-view-finetune] failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
