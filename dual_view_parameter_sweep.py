#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Serial, CPU-only dual-view experiments. New schema; no legacy review adapter.

The supervisor imports only stdlib. Torch lives in one disposable child at a
time, so the supervisor does not retain a second model/runtime in scarce RAM.
Real data requires the frozen classification contract and --allow-training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

FORMAT = "birdsvision-dual-view-sweep-v1"
EXAMPLE = {"format": FORMAT, "trainable": "head", "epochs": 3,
           "learning_rates": [0.00003, 0.0001, 0.0003], "weight_decays": [0.01, 0.05],
           "seeds": [20260907, 20260908], "train_per_class": 2,
           "validation_per_class": 2, "subset_seed": 20260907}
SELECTION = "mean_seed_macro_top1_desc_then_nll_asc_then_ece_asc"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    os.replace(temporary, path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_plan(plan):
    if set(plan) != set(EXAMPLE) or plan["format"] != FORMAT:
        raise ValueError("plan requires exactly the documented v1 fields")
    if plan["trainable"] not in ("head", "all"):
        raise ValueError("trainable must be head or all")
    for key in ("epochs", "train_per_class", "validation_per_class"):
        if type(plan[key]) is not int or plan[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(plan["subset_seed"]) is not int or plan["subset_seed"] < 0:
        raise ValueError("invalid subset_seed")
    for key in ("learning_rates", "weight_decays", "seeds"):
        values = plan[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty list")
        for value in values:
            if key == "seeds":
                valid = type(value) is int and 0 <= value < 2**32
            else:
                valid = (type(value) in (int, float) and math.isfinite(value)
                         and (value > 0 if key == "learning_rates" else value >= 0))
            if not valid:
                raise ValueError(f"invalid {key}")
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate {key}")
    return plan


def trials(plan):
    result = []
    for group, (lr, wd) in enumerate(itertools.product(plan["learning_rates"], plan["weight_decays"])):
        for seed in plan["seeds"]:
            result.append({"trial_id": f"trial-{len(result) + 1:04d}", "group_id": f"group-{group + 1:04d}",
                           "learning_rate": lr, "weight_decay": wd, "seed": seed})
    return result


def select_records(records, plan):
    groups = {}
    for row in records:
        if row["split"] not in ("train", "validation"):
            raise ValueError("final_test is sealed")
        groups.setdefault((row["split"], row["model_class_id"]), []).append(row)
    selected = set()
    for (split, _), rows in groups.items():
        ordered = sorted(rows, key=lambda r: (hashlib.sha256(canonical(
            [plan["subset_seed"], r["record_id"]])).hexdigest(), r["record_id"]))
        selected.update(r["record_id"] for r in ordered[:plan[f"{split}_per_class"]])
    # Preserve manifest order; never split one parent's views or truncate boxes.
    return [row for row in records if row["record_id"] in selected]


def configure_trainable(model, mode):
    if mode not in ("head", "all"):
        raise ValueError("unknown trainable mode")
    for parameter in model.parameters():
        parameter.requires_grad_(mode == "all")
    if mode == "head":
        for parameter in model.backbone.classifier[2].parameters():
            parameter.requires_grad_(True)
        # The engine calls model.train() each epoch. Keep frozen stochastic
        # layers in evaluation mode while retaining the ordinary backbone keys.
        def frozen_features(module, _inputs):
            module.backbone.eval()
            module.backbone.classifier[2].train(module.training)
        model.register_forward_pre_hook(frozen_features)


def evaluate(model, loader, prediction_path):
    import torch
    import dual_view_classifier_train as train
    total = correct = top3 = 0
    nll = brier = confidence_sum = correct_confidence = wrong_confidence = 0.0
    bins = [[0, 0.0, 0] for _ in range(15)]
    classes = {}
    model.eval()
    temporary = prediction_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream, torch.no_grad():
        for batch in loader:
            outputs = train.forward_batch(model, batch, torch.device("cpu"))
            logp = outputs["fused"].double().log_softmax(1)
            if not torch.isfinite(logp).all():
                raise ValueError("nonfinite validation logits")
            probabilities = logp.exp()
            confidence, predictions = probabilities.max(1)
            top = probabilities.topk(min(3, probabilities.shape[1]), dim=1).indices
            for index, record in enumerate(batch["records"]):
                label = int(batch["labels"][index])
                predicted = int(predictions[index])
                conf = float(confidence[index])
                hit = int(predicted == label)
                total += 1
                correct += hit
                top3 += int(label in top[index].tolist())
                nll -= float(logp[index, label])
                brier += float(probabilities[index].square().sum() - 2 * probabilities[index, label] + 1)
                confidence_sum += conf
                correct_confidence += conf * hit
                wrong_confidence += conf * (1 - hit)
                bucket = bins[min(14, int(conf * 15))]
                bucket[0] += 1
                bucket[1] += conf
                bucket[2] += hit
                counts = classes.setdefault(str(label), [0, 0])
                counts[0] += 1
                counts[1] += hit
                stream.write(canonical({"record_id": record["record_id"], "label": label,
                    "prediction": predicted, "confidence": conf, "correct": bool(hit),
                    "true_class_probability": float(probabilities[index, label]),
                    "full_prediction": int(outputs["full"][index].argmax()),
                    "crop_prediction": int(outputs["crop"][index].argmax())}).decode() + "\n")
    if not total:
        raise ValueError("empty validation")
    os.replace(temporary, prediction_path)
    return {"parents": total, "top1": correct / total, "top3": top3 / total,
            "macro_top1": statistics.mean(hit / count for count, hit in classes.values()),
            "nll": nll / total, "brier": brier / total,
            "ece": sum(abs(hit - conf) for count, conf, hit in bins) / total,
            "mean_confidence": confidence_sum / total,
            "correct_mean_confidence": correct_confidence / correct if correct else None,
            "wrong_mean_confidence": wrong_confidence / (total - correct) if total != correct else None,
            "calibration_bins_count_confidence_sum_correct": bins, "class_counts_correct": classes}


def rank_results(plan, results, *, synthetic):
    expected = trials(plan)
    by_id = {r["trial"]["trial_id"]: r for r in results}
    if len(by_id) != len(results):
        raise ValueError("duplicate trial results")
    ranking = []
    for group_id in dict.fromkeys(t["group_id"] for t in expected):
        members = [t for t in expected if t["group_id"] == group_id]
        if any(t["trial_id"] not in by_id for t in members):
            continue
        rows = [by_id[t["trial_id"]] for t in members]
        if any(row["trial"] != t for row, t in zip(rows, members, strict=True)):
            raise ValueError("trial identity mismatch")
        metrics = {key: statistics.mean(r["validation"][key] for r in rows)
                   for key in ("macro_top1", "top1", "top3", "nll", "ece", "brier", "mean_confidence")}
        if not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError("nonfinite comparison metrics")
        ranking.append({"group_id": group_id, "learning_rate": members[0]["learning_rate"],
                        "weight_decay": members[0]["weight_decay"], "seeds": plan["seeds"],
                        "mean": metrics, "macro_top1_seed_std": statistics.pstdev(
                            r["validation"]["macro_top1"] for r in rows)})
    ranking.sort(key=lambda r: (-r["mean"]["macro_top1"], r["mean"]["nll"], r["mean"]["ece"], r["group_id"]))
    complete = len(results) == len(expected)
    return {"selection_rule": SELECTION, "complete": complete, "synthetic": synthetic,
            "completed_trials": len(results), "expected_trials": len(expected), "ranking": ranking,
            "best_parameters": ranking[0] if complete and ranking and not synthetic else None,
            "scope": "selected_validation_subset_only_not_final_test_or_production_acceptance"}


def memory_sample(pid):
    """Linux RSS/HWM and host MemAvailable in bytes; no dependency installation."""
    if not sys.platform.startswith("linux"):
        return {"rss_bytes": None, "hwm_bytes": None, "available_bytes": None}
    def fields(path):
        return {line.split(":", 1)[0]: line.split(":", 1)[1].split()
                for line in Path(path).read_text().splitlines() if ":" in line}
    try:
        status = fields(f"/proc/{pid}/status")
        mem = fields("/proc/meminfo")
        return {"rss_bytes": int(status["VmRSS"][0]) * 1024,
                "hwm_bytes": int(status["VmHWM"][0]) * 1024,
                "available_bytes": int(mem["MemAvailable"][0]) * 1024}
    except (FileNotFoundError, ProcessLookupError, KeyError):
        return {"rss_bytes": None, "hwm_bytes": None, "available_bytes": None}


def resource_reason(sample, args):
    if sample["rss_bytes"] is not None and sample["rss_bytes"] > args.max_rss_mib * 1024**2:
        return "worker_rss_limit"
    if sample["available_bytes"] is not None and sample["available_bytes"] < args.min_available_mib * 1024**2:
        return "host_available_memory_floor"
    return None


def monitored(command, directory, args):
    stamp = str(time.time_ns())
    started = time.monotonic()
    reason = None
    stopped_at = None
    peak = 0
    hwm = 0
    minimum = None
    count = 0
    phase_peaks = {}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads), NUMEXPR_NUM_THREADS=str(args.threads))
    sample = memory_sample(os.getpid())
    refusal = None
    if sample["available_bytes"] is not None and sample["available_bytes"] < args.min_available_mib * 1024**2:
        refusal = "host_available_memory_floor_before_launch"
    if shutil.disk_usage(directory).free < args.min_free_disk_mib * 1024**2:
        refusal = refusal or "free_disk_floor_before_launch"
    if refusal:
        report = {"exit_code": 125, "stop_reason": refusal, "elapsed_seconds": 0,
                  "peak_sampled_rss_bytes": None, "peak_observed_hwm_bytes": None,
                  "min_host_available_bytes": sample["available_bytes"], "memory_samples": 0,
                  "sample_seconds": args.sample_seconds, "phase_peak_sampled_rss_bytes": {},
                  "monitor": "linux_proc" if sys.platform.startswith("linux") else "unavailable",
                  "hard_memory_limit": False, "log": None, "trace": None}
        save(directory / f"resources-{stamp}.json", report)
        return report
    with (directory / f"attempt-{stamp}.log").open("wb") as log, (directory / f"memory-{stamp}.jsonl").open("wb") as trace:
        save(directory / "phase.json", {"phase": "worker_start"})
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        def request_stop(signum, _frame):
            nonlocal reason, stopped_at
            if reason is None:
                reason, stopped_at = f"signal_{signum}", time.monotonic()
                child.terminate()
        previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            while child.poll() is None:
                sample = memory_sample(child.pid)
                elapsed = time.monotonic() - started
                phase = read(directory / "phase.json")["phase"]
                trace.write(canonical({"elapsed_seconds": elapsed, "phase": phase, **sample}) + b"\n")
                trace.flush()
                if sample["rss_bytes"] is not None:
                    count += 1
                    peak = max(peak, sample["rss_bytes"])
                    hwm = max(hwm, sample["hwm_bytes"] or 0)
                    phase_peaks[phase] = max(phase_peaks.get(phase, 0), sample["rss_bytes"])
                if sample["available_bytes"] is not None:
                    minimum = min(minimum if minimum is not None else sample["available_bytes"], sample["available_bytes"])
                if reason is None:
                    reason = resource_reason(sample, args)
                    if args.max_trial_minutes and elapsed >= args.max_trial_minutes * 60:
                        reason = reason or "trial_time_budget"
                    if reason:
                        stopped_at = time.monotonic()
                        child.terminate()
                if stopped_at is not None and time.monotonic() - stopped_at > 60:
                    child.kill()
                time.sleep(args.sample_seconds)
            code = child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    report = {"exit_code": code, "stop_reason": reason, "elapsed_seconds": time.monotonic() - started,
              "peak_sampled_rss_bytes": peak if count else None,
              "peak_observed_hwm_bytes": hwm if count else None, "min_host_available_bytes": minimum,
              "memory_samples": count, "sample_seconds": args.sample_seconds,
              "phase_peak_sampled_rss_bytes": phase_peaks,
              "monitor": "linux_proc" if sys.platform.startswith("linux") else "unavailable",
              "hard_memory_limit": False, "log": f"attempt-{stamp}.log", "trace": f"memory-{stamp}.jsonl"}
    save(directory / f"resources-{stamp}.json", report)
    return report


def prepare(args, plan):
    import torch
    import dual_view_training_data as data
    import dual_view_classifier_train as train
    torch.set_num_threads(args.threads)
    if args.synthetic_smoke:
        root = args.output_dir / "synthetic-data"
        if not root.exists():
            data.synthetic_bundle(root)
        contract_path = root / "data_contract.json"
        contract_hash = sha(contract_path)
    else:
        root, contract_path, contract_hash = args.project_root, args.data_contract, args.data_contract_sha256
    bundle = data.load_bundle(root, contract_path, contract_hash, synthetic=args.synthetic_smoke)
    bundle.records = select_records(bundle.records, plan)
    for split in ("train", "validation"):
        data.ViewBudgetBatchSampler([r for r in bundle.records if r["split"] == split], args.max_views)
    initialization = {"kind": "synthetic_untrained"}
    if not args.synthetic_smoke:
        for path, expected in ((args.init_checkpoint, args.init_checkpoint_sha256),
                               (args.init_class_map, args.init_class_map_sha256)):
            data.require_hash(expected)
            if sha(path) != expected:
                raise ValueError("initialization SHA-256 mismatch")
        initialization = {"checkpoint_sha256": args.init_checkpoint_sha256,
                          "source_class_map_sha256": args.init_class_map_sha256}
    context = {"format": FORMAT, "plan": plan, "synthetic": args.synthetic_smoke,
               "data_contract_sha256": contract_hash, "data_contract": bundle.contract,
               "record_ids": [r["record_id"] for r in bundle.records],
               "initialization": initialization, "environment": train.runtime_contract(torch.device("cpu")),
               "sweep_source_sha256": sha(__file__), "selection_rule": SELECTION,
               "max_views": args.max_views, "threads": args.threads,
               "guards": {k: getattr(args, k) for k in ("max_rss_mib", "min_available_mib", "min_free_disk_mib",
                                                        "max_trial_minutes", "sample_seconds")},
               "coverage": {split: {"parents": sum(r["split"] == split for r in bundle.records),
                    "present_classes": len({r["model_class_id"] for r in bundle.records if r["split"] == split}),
                    "class_parent_counts": {str(class_id): sum(r["split"] == split and r["model_class_id"] == class_id
                        for r in bundle.records) for class_id in sorted({r["model_class_id"] for r in bundle.records if r["split"] == split})}}
                    for split in ("train", "validation")}}
    path = args.output_dir / "sweep_contract.json"
    if path.exists() and read(path) != context:
        raise ValueError("sweep resume contract mismatch")
    if not path.exists():
        save(path, context)
    return bundle, context


def worker(args, plan):
    if args.internal_stage == "trial" and args.trial_id not in {t["trial_id"] for t in trials(plan)}:
        raise ValueError("unknown exact trial_id")
    if hasattr(os, "nice"):
        os.nice(10)
    directory = args.output_dir if args.internal_stage == "prepare" else args.output_dir / args.trial_id
    save(directory / "phase.json", {"phase": "initialization_and_preflight"})
    bundle, context = prepare(args, plan)
    if args.internal_stage == "prepare":
        print(json.dumps({"status": "preflight_complete", "coverage": context["coverage"]}), flush=True)
        return 0
    if not args.allow_training and not args.synthetic_smoke:
        raise ValueError("real experiments require --allow-training")
    import torch
    import dual_view_classifier_train as train
    trial = next(t for t in trials(plan) if t["trial_id"] == args.trial_id)
    train.seed_everything(trial["seed"])
    model = train.StableCUBModel(num_classes=len(bundle.classes), use_pretrained=False)
    if not args.synthetic_smoke:
        train.initialize_exact(model, args.init_checkpoint, args.init_checkpoint_sha256,
                               args.init_class_map, args.init_class_map_sha256,
                               [c["class_key"] for c in bundle.classes])
    configure_trainable(model, plan["trainable"])
    transforms = train.make_transforms(model)
    settings = {"max_views": args.max_views, "epochs": plan["epochs"], "num_workers": 0,
                "learning_rate": trial["learning_rate"], "weight_decay": trial["weight_decay"],
                "seed": trial["seed"], "alpha": 0.5, "amp": False, "patience": 0,
                "initialization": context["initialization"], "sweep_contract_sha256": sha(args.output_dir / "sweep_contract.json"),
                "trainable": plan["trainable"], "selected_record_ids": context["record_ids"]}
    contract = train.build_run_contract(bundle, settings, transforms[2], transforms[:2], context["environment"])
    run = directory / "training"
    checkpoints = [run / name for name in ("latest.checkpoint.pth", "interrupted.checkpoint.pth") if (run / name).is_file()]
    resume = max(checkpoints, key=lambda p: p.stat().st_mtime_ns) if checkpoints else None
    save(directory / "phase.json", {"phase": "training_with_epoch_validation_and_checkpoints"})
    code = train.run_epochs(bundle, model, transforms[:2], contract, run, resume_path=resume)
    if code:
        return code
    # Fixed epoch budget for every trial; evaluate the FINAL checkpoint, not
    # an opportunistically selected validation peak from unequal search budgets.
    save(directory / "phase.json", {"phase": "final_validation"})
    validation = evaluate(model, train.make_loader(bundle, "validation", transforms[1], settings, plan["epochs"]),
                          directory / "validation_predictions.jsonl")
    save(directory / "result.json", {"trial": trial, "sweep_contract_sha256": settings["sweep_contract_sha256"],
         "validation": validation, "epochs": plan["epochs"], "trainable": plan["trainable"],
         "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
         "total_parameters": sum(p.numel() for p in model.parameters()),
         "checkpoint": "training/latest.checkpoint.pth", "checkpoint_sha256": sha(run / "latest.checkpoint.pth")})
    return 0


def collect(args, plan):
    results = []
    all_trials = []
    contract_hash = sha(args.output_dir / "sweep_contract.json")
    for trial in trials(plan):
        directory = args.output_dir / trial["trial_id"]
        path = directory / "result.json"
        resources = [read(p) for p in sorted(directory.glob("resources-*.json"))]
        peaks = [r["peak_observed_hwm_bytes"] for r in resources if r["peak_observed_hwm_bytes"] is not None]
        training_peaks = [r["phase_peak_sampled_rss_bytes"].get("training_with_epoch_validation_and_checkpoints") for r in resources]
        resource_summary = {"resource_attempts": resources,
            "peak_observed_hwm_bytes": max(peaks) if peaks else None,
            "training_peak_sampled_rss_bytes": max((v for v in training_peaks if v is not None), default=None),
            "elapsed_seconds_all_attempts": sum(r["elapsed_seconds"] for r in resources)}
        entry = {"trial": trial, "status": "unfinished" if resources else "pending", **resource_summary}
        all_trials.append(entry)
        if path.exists():
            result = read(path)
            if result["trial"] != trial or result["sweep_contract_sha256"] != contract_hash:
                raise ValueError("completed result contract mismatch")
            if result["checkpoint"] != "training/latest.checkpoint.pth" or sha(directory / result["checkpoint"]) != result["checkpoint_sha256"]:
                raise ValueError("completed checkpoint SHA-256 mismatch")
            if not resources or resources[-1]["exit_code"] != 0 or resources[-1]["stop_reason"] is not None:
                continue
            result.update(resource_summary)
            entry.update(result, status="complete")
            results.append(result)
    summary = rank_results(plan, results, synthetic=args.synthetic_smoke)
    summary["results"] = results
    summary["trials"] = all_trials
    save(args.output_dir / "comparison.json", summary)
    temporary = args.output_dir / "comparison.csv.tmp"
    fields = ["trial_id", "group_id", "learning_rate", "weight_decay", "seed", "status", "epochs", "trainable",
              "top1", "macro_top1", "top3", "nll", "ece", "brier", "mean_confidence",
              "correct_mean_confidence", "wrong_mean_confidence", "peak_observed_hwm_bytes",
              "training_peak_sampled_rss_bytes", "elapsed_seconds_all_attempts"]
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in all_trials:
            flat = {"epochs": plan["epochs"], "trainable": plan["trainable"],
                    **r, **r["trial"], **r.get("validation", {})}
            writer.writerow({key: flat.get(key) for key in fields})
    os.replace(temporary, args.output_dir / "comparison.csv")
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-example-plan", type=Path)
    p.add_argument("--plan", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--synthetic-smoke", action="store_true")
    p.add_argument("--allow-training", action="store_true")
    p.add_argument("--resume-sweep", action="store_true")
    p.add_argument("--max-views", type=int)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--max-rss-mib", type=int, default=1024)
    p.add_argument("--min-available-mib", type=int, default=768)
    p.add_argument("--min-free-disk-mib", type=int, default=2048)
    p.add_argument("--sample-seconds", type=float, default=0.5)
    p.add_argument("--max-trial-minutes", type=float, default=0)
    for key in ("project-root", "data-contract", "init-checkpoint", "init-class-map"):
        p.add_argument("--" + key, type=Path)
    for key in ("data-contract-sha256", "init-checkpoint-sha256", "init-class-map-sha256"):
        p.add_argument("--" + key)
    p.add_argument("--internal-stage", choices=("prepare", "trial"), help=argparse.SUPPRESS)
    p.add_argument("--trial-id", help=argparse.SUPPRESS)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    try:
        if args.write_example_plan:
            with args.write_example_plan.open("xb") as stream:
                stream.write(canonical(EXAMPLE) + b"\n")
            return 0
        if args.output_dir is None or (args.plan is None and not args.synthetic_smoke):
            raise ValueError("explicit output-dir and plan required")
        plan = validate_plan(read(args.plan) if args.plan else {
            **EXAMPLE, "epochs": 1, "learning_rates": [0.0001, 0.0003], "weight_decays": [0.05], "seeds": [20260907]})
        if args.synthetic_smoke and (args.plan or args.allow_training or any(
                getattr(args, k) is not None for k in ("project_root", "data_contract", "data_contract_sha256",
                    "init_checkpoint", "init_checkpoint_sha256", "init_class_map", "init_class_map_sha256"))):
            raise ValueError("synthetic smoke uses generated data and a fixed plan only")
        if args.max_views is None:
            if args.synthetic_smoke:
                args.max_views = 3
            else:
                raise ValueError("explicit max-views required")
        for key in ("threads", "max_views", "max_rss_mib", "min_available_mib", "min_free_disk_mib"):
            if getattr(args, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not math.isfinite(args.sample_seconds) or not 0.1 <= args.sample_seconds <= 5:
            raise ValueError("sample-seconds must be in [0.1, 5]")
        if not math.isfinite(args.max_trial_minutes) or args.max_trial_minutes < 0:
            raise ValueError("invalid max-trial-minutes")
        if not args.synthetic_smoke and not all(getattr(args, key) for key in (
                "project_root", "data_contract", "data_contract_sha256", "init_checkpoint", "init_checkpoint_sha256",
                "init_class_map", "init_class_map_sha256")):
            raise ValueError("real experiments require exact data and initialization paths/hashes")
        if not args.synthetic_smoke and not sys.platform.startswith("linux"):
            raise ValueError("real CPU sweep requires Linux /proc memory monitoring")
        args.output_dir = args.output_dir.resolve()
        if args.internal_stage:
            return worker(args, plan)
        args.output_dir.mkdir(parents=True, exist_ok=args.resume_sweep)
        lock = args.output_dir / "sweep.lock"
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        try:
            command = [sys.executable, str(Path(__file__).resolve()), *argv]
            report = monitored([*command, "--internal-stage", "prepare"], args.output_dir, args)
            if report["exit_code"] or report["stop_reason"]:
                return 1
            if not args.synthetic_smoke and not args.allow_training:
                print("Preflight complete; add --resume-sweep --allow-training to execute this exact experiment.")
                return 0
            summary = collect(args, plan)
            for trial in trials(plan):
                directory = args.output_dir / trial["trial_id"]
                directory.mkdir(exist_ok=True)
                if trial["trial_id"] in {r["trial"]["trial_id"] for r in summary["results"]}:
                    continue
                print(f"Starting {trial['trial_id']}: {trial}", flush=True)
                report = monitored([*command, "--internal-stage", "trial", "--trial-id", trial["trial_id"]], directory, args)
                summary = collect(args, plan)
                print(json.dumps({"trial": trial["trial_id"], "resources": report,
                                  "completed_trials": summary["completed_trials"]}), flush=True)
                if report["exit_code"] or report["stop_reason"]:
                    return 1
            print(json.dumps({"comparison": str(args.output_dir / "comparison.json"),
                              "best_parameters": collect(args, plan)["best_parameters"]}), flush=True)
            return 0
        finally:
            lock.unlink()
    except Exception as exc:
        print(f"[dual-view-sweep] failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
