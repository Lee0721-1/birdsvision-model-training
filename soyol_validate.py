# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Compare saved SOYOL checkpoints on A-tier validation with one-to-many NMS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from soyol_dataset import verify
from soyol_output_policy import with_soyol_max_det


def run(dataset, weights, output):
    verified = verify(dataset)
    if output.exists():
        raise FileExistsError(output)
    results = {}
    for name, path in weights.items():
        model = YOLO(str(path))
        if model.task != "detect" or model.names != {0: "bird"}:
            raise ValueError("invalid SOYOL checkpoint")
        model.model.end2end = False
        options = with_soyol_max_det({"data": str(dataset / "soyol_detect.yaml"),
                                      "imgsz": 640, "batch": 16, "conf": 0.001,
                                      "iou": 0.7, "device": 0, "plots": False,
                                      "project": str(output.parent), "name": output.stem + "_" + name})
        metrics = model.val(**options)
        if model.model.end2end:
            raise ValueError("validation did not use the NMS branch")
        results[name] = {"mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
                         "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                         "weight_path": str(path)}
    selected = max(results, key=lambda key: results[key]["mAP50_95"])
    report = {"format": "birdsvision-soyol-a-nms-validation-v1", "dataset_verification": verified,
              "postprocess": "one_to_many_branch_then_nms", "max_det": 10,
              "results": results, "selected_checkpoint": selected,
              "final_test_read_or_written": False}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--best", type=Path, required=True)
    parser.add_argument("--last", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset.resolve(), {"best": args.best.resolve(), "last": args.last.resolve()}, args.output.resolve()), ensure_ascii=False, indent=2))
