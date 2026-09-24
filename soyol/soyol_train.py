# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Run bounded SOYOL A-tier Detect GPU smoke or an explicitly configured train run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from soyol.soyol_dataset import digest, verify
from soyol.soyol_output_policy import with_soyol_max_det


OFFICIAL_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
OFFICIAL_URL = "https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26n.pt"


def run(args):
    dataset, base, project = args.dataset.resolve(), args.base.resolve(), args.project.resolve()
    verified = verify(dataset)
    if digest(base) != OFFICIAL_SHA256:
        raise ValueError("base weight differs from published Ultralytics YOLO26n Detect file")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    model = YOLO(str(base))
    if model.task != "detect" or type(model.model).__name__ != "DetectionModel" or len(model.names) != 80:
        raise ValueError("base must be COCO YOLO26n Detect")
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise ValueError("invalid training configuration")
    if args.name in ("", ".", "..") or "/" in args.name or "\\" in args.name:
        raise ValueError("invalid run name")
    run_dir = project / args.name
    if run_dir.exists():
        raise FileExistsError(run_dir)
    project.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    settings = with_soyol_max_det({
        "data": str(dataset / "soyol_detect.yaml"), "epochs": args.epochs,
        "imgsz": args.imgsz, "batch": args.batch, "workers": args.workers,
        "device": 0, "project": str(project), "name": args.name,
        "exist_ok": True, "seed": 23, "deterministic": True,
        "plots": False, "verbose": True,
        "fraction": 0.02 if args.smoke else 1.0,
        "val": not args.smoke,
    })
    contract = {
        "format": "birdsvision-soyol-a-detect-run-v1", "status": "started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "gpu_smoke" if args.smoke else "formal_train",
        "dataset_contract_sha256": digest(dataset / "data_contract.json"),
        "dataset_verification": verified,
        "base_weight_path": str(base), "base_weight_sha256": digest(base),
        "base_weight_official_source": OFFICIAL_URL,
        "runtime": {"torch": torch.__version__, "ultralytics": ultralytics.__version__, "cuda_device": torch.cuda.get_device_name(0)},
        "arguments": settings, "tylo_checkpoint_loaded": False, "final_test_read_or_written": False,
    }
    contract_path = run_dir / "soyol_run_contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    try:
        result = model.train(**settings)
    except BaseException as error:
        contract["status"] = "failed"
        contract["error"] = repr(error)
        contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        raise
    contract["status"] = "completed"
    contract["result_save_dir"] = str(result.save_dir)
    contract["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2, default=str))
