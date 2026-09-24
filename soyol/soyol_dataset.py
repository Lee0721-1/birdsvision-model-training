# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Verify an externally supplied SOYOL A-tier Detect export before training."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit


SPLIT_SEED = "soyol-a-detect-train-validation-20260923-v1"
ALLOWED_LICENSES = {"cc-by", "cc0", "cc0-1.0"}
HUMAN_FINAL_PROVENANCE = {"human_approved_teacher_box", "human_manual_box"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def split_for(group: str) -> str:
    value = int(hashlib.sha256(f"{SPLIT_SEED}\0{group}".encode()).hexdigest()[:16], 16)
    return "validation" if value % 5 == 0 else "train"


def render_label(instances: list[dict]) -> str:
    lines = []
    for instance in instances:
        box = instance["bbox_xyxy_normalized"]
        if (len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                 or not math.isfinite(v) for v in box)):
            raise ValueError("invalid normalized bird box")
        left, top, right, bottom = box
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError("invalid normalized bird box")
        lines.append("0 " + " ".join(f"{v:.10f}" for v in (
            (left + right) / 2, (top + bottom) / 2, right - left, bottom - top,
        )))
    return "\n".join(lines) + ("\n" if lines else "")


def verify(directory: Path) -> dict:
    """Bind files, labels, group splits and the private selection without publishing them."""
    directory = directory.resolve()
    contract = json.loads((directory / "data_contract.json").read_text(encoding="utf-8"))
    if contract["format"] != "birdsvision-soyol-a-detect-export-v1":
        raise ValueError("unexpected dataset contract")
    if contract["final_test_read_or_written"] is not False:
        raise ValueError("final_test is outside this training export")
    manifest = directory / "export_manifest.jsonl"
    yaml = directory / "soyol_detect.yaml"
    if digest(manifest) != contract["manifest_sha256"] or digest(yaml) != contract["yaml_sha256"]:
        raise ValueError("manifest or yaml changed")
    if digest(Path(contract["selection_path"])) != contract["selection_sha256"]:
        raise ValueError("source selection changed")
    seen, group_splits, image_splits = set(), {}, {}
    counts = Counter()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        record_id, split, group = row["record_id"], row["split"], row["soyol_group_id"]
        page = urlsplit(row["source_page_url"])
        if (row["license_code"] not in ALLOWED_LICENSES or not row["attribution"]
                or page.scheme != "https" or not page.hostname
                or row["source_split_for_audit_only"] not in {"train", "validation"}):
            raise ValueError("A-tier rights or source evidence missing")
        if record_id in seen or split != split_for(group):
            raise ValueError("record or split mismatch")
        seen.add(record_id)
        if group_splits.setdefault(group, split) != split:
            raise ValueError("group leakage")
        if image_splits.setdefault(row["source_image_sha256"], split) != split:
            raise ValueError("image leakage")
        source = Path(row["source_image_path"])
        image = (directory / row["copied_image_path"]).resolve()
        label = (directory / row["label_path"]).resolve()
        if not image.is_relative_to(directory) or not label.is_relative_to(directory):
            raise ValueError("export path escapes dataset")
        if image.is_symlink() or label.is_symlink():
            raise ValueError("link in export")
        if (digest(source) != row["source_image_sha256"]
                or digest(image) != row["copied_image_sha256"]
                or row["source_image_sha256"] != row["copied_image_sha256"]):
            raise ValueError("image mismatch")
        instances = row["instances"]
        if any(instance["provenance"] not in HUMAN_FINAL_PROVENANCE for instance in instances):
            raise ValueError("box is not human-confirmed")
        if not instances and row["source_scope"] != "full_v1_explicit_human_no_bird":
            raise ValueError("empty label lacks explicit human no-bird review")
        expected = render_label(instances)
        if label.read_text(encoding="utf-8") != expected or digest(label) != row["label_sha256"]:
            raise ValueError("label mismatch")
        counts[f"{split}_parents"] += 1
        counts[f"{split}_boxes"] += len(instances)
        counts[f"{split}_no_bird"] += not instances
    for split in ("train", "validation"):
        for field in ("parents", "boxes", "no_bird"):
            if counts[f"{split}_{field}"] != contract["counts"][split][field]:
                raise ValueError("count mismatch")
    return {"status": "verified", "counts": dict(counts), "groups": len(group_splits)}
