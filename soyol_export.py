# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Export an external, reviewed A-tier SOYOL selection into Detect train/val files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from soyol_dataset import (ALLOWED_LICENSES, HUMAN_FINAL_PROVENANCE, SPLIT_SEED,
                           digest, render_label, split_for, verify)


def export(selection: Path, output: Path) -> dict:
    selection, output = selection.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report_path = selection.parent / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report["selection_sha256"] != digest(selection)
            or report["final_test_read_or_written"] is not False):
        raise ValueError("selection report mismatch or final_test use")
    rows = [json.loads(line) for line in selection.read_text(encoding="utf-8").splitlines()]
    if not rows or len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("empty selection or duplicate record IDs")
    group_splits, content_splits = {}, {}
    counters = defaultdict(Counter)
    output.mkdir(parents=True)
    emitted = []
    for row in rows:
        record_id = row["record_id"]
        if row["license_code"] not in ALLOWED_LICENSES or not row["attribution"]:
            raise ValueError(f"A-tier license or attribution missing: {record_id}")
        parsed = urlsplit(row["source_page_url"])
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError(f"source page missing: {record_id}")
        if row["source_split_for_audit_only"] not in {"train", "validation"}:
            raise ValueError("selection contains an unsupported or sealed split")
        instances = row["instances"]
        if instances:
            if any(item["provenance"] not in HUMAN_FINAL_PROVENANCE for item in instances):
                raise ValueError(f"box is not human-confirmed: {record_id}")
        elif row["source_scope"] != "full_v1_explicit_human_no_bird":
            raise ValueError(f"empty image is not human-confirmed no-bird: {record_id}")
        if row["detect_label"] != render_label(instances):
            raise ValueError(f"Detect label differs from final boxes: {record_id}")
        group = row["soyol_group_id"]
        split = group_splits.setdefault(group, split_for(group))
        if content_splits.setdefault(row["source_image_sha256"], split) != split:
            raise ValueError(f"image crosses train/validation: {record_id}")
        source = Path(row["source_image_path"])
        if not source.is_file() or digest(source) != row["source_image_sha256"]:
            raise ValueError(f"source image mismatch: {record_id}")
        stem = hashlib.sha256(record_id.encode()).hexdigest()
        folder = "val" if split == "validation" else "train"
        image = output / "images" / folder / (stem + source.suffix.lower())
        label = output / "labels" / folder / (stem + ".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image)
        if image.is_symlink() or os.path.samefile(source, image) or digest(image) != row["source_image_sha256"]:
            raise ValueError(f"copied image mismatch: {record_id}")
        label.write_text(row["detect_label"], encoding="utf-8", newline="\n")
        if digest(source) != row["source_image_sha256"]:
            raise ValueError(f"source changed during export: {record_id}")
        counters[split]["parents"] += 1
        counters[split]["boxes"] += len(instances)
        counters[split]["no_bird"] += not instances
        emitted.append({key: value for key, value in row.items() if key != "detect_label"} | {
            "split": split,
            "copied_image_path": image.relative_to(output).as_posix(),
            "copied_image_sha256": digest(image),
            "label_path": label.relative_to(output).as_posix(),
            "label_sha256": digest(label),
        })
    if any(counters[split]["parents"] == 0 for split in ("train", "validation")):
        raise ValueError("train and validation must both be non-empty")
    manifest = output / "export_manifest.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as stream:
        for row in emitted:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    yaml = output / "soyol_detect.yaml"
    yaml.write_text(f"path: '{output.as_posix()}'\ntrain: images/train\nval: images/val\nnames:\n  0: bird\n",
                    encoding="utf-8", newline="\n")
    contract = {
        "format": "birdsvision-soyol-a-detect-export-v1",
        "status": "export_completed_not_training_authorized",
        "selection_path": str(selection), "selection_sha256": digest(selection),
        "selection_report_sha256": digest(report_path),
        "split_policy": {"seed": SPLIT_SEED,
                         "validation_modulo": "sha256(seed\\0group) first64 % 5 == 0",
                         "visual_group_rule": report["visual_group_rule"]},
        "counts": {name: dict(values) for name, values in counters.items()},
        "groups_by_split": dict(Counter(group_splits.values())),
        "manifest_sha256": digest(manifest), "yaml_sha256": digest(yaml),
        "final_test_read_or_written": False,
    }
    (output / "data_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verify(output)
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.selection, args.output), ensure_ascii=False, indent=2))
