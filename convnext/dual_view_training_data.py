# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Versioned parent/full/crop input contract; never reads live review workspaces.

This is a NEW classification schema, not either review tool's state schema.
An independent, reviewed export must provide train/validation-only JSONL files,
an exclusion JSONL, a class table and a hash-bound contract. No export/freeze of
real data is performed here. See ``synthetic_bundle`` for a complete executable
schema example. Source images are decoded from verified bytes and never written.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

FORMAT = "birdsvision-dual-view-data-v1"
ALLOWED_SPLITS = ("train", "validation")
CROP_RULE = {
    "crop_expansion_fraction_per_side": 0.15,
    "clip_to_image_boundary": True,
    "pixel_rounding": "floor_left_top_ceil_right_bottom",
}
ZERO_RULE = "exclude_final_zero_boxes_from_supervised_classification"
ORIGINS = ("ground_truth", "original_model_box", "manual_box")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError("SHA-256 must be exactly 64 lowercase hexadecimal characters")
    return value


def require_split(split):
    if split not in ALLOWED_SPLITS:
        raise ValueError("ordinary training accepts only train/validation; final_test is sealed")


def relative_file(root: Path, value: str) -> Path:
    # A new portable schema explicitly uses POSIX relative paths; no repair or
    # case/separator normalization of legacy identifiers is attempted.
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("storage_path must be a portable relative POSIX path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("invalid relative path components")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive:
        raise ValueError("absolute storage_path is forbidden")
    root = root.resolve()
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("source symlinks/junctions are forbidden")
    if not path.resolve().is_relative_to(root):
        raise ValueError("path escapes project root")
    return path


def verified_bytes(path: Path, expected_sha256: str, byte_size=None) -> bytes:
    require_hash(expected_sha256)
    data = path.read_bytes()
    if byte_size is not None and len(data) != byte_size:
        raise ValueError(f"byte_size mismatch: {path}")
    if digest(data) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return data


def normalized_box(value):
    if not isinstance(value, list) or len(value) != 4 or any(
        type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
        for x in value
    ):
        raise ValueError("bbox must contain four finite coordinates in [0, 1]")
    x1, y1, x2, y2 = value
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox must have positive area")
    return value


def crop_geometry(bbox, width, height):
    x1, y1, x2, y2 = normalized_box(bbox)
    dx, dy = (x2 - x1) * 0.15, (y2 - y1) * 0.15
    expanded = [max(0.0, x1 - dx), max(0.0, y1 - dy),
                min(1.0, x2 + dx), min(1.0, y2 + dy)]
    pixels = [math.floor(expanded[0] * width), math.floor(expanded[1] * height),
              math.ceil(expanded[2] * width), math.ceil(expanded[3] * height)]
    return expanded, pixels


def supervision(record, classes):
    """Return exact/coarse output ids without inventing one fine-grained label."""
    value = record.get("supervision")
    if value is None:
        class_id = record["model_class_id"]
        if type(class_id) is not int or not 0 <= class_id < len(classes):
            raise ValueError("model_class_id must be an exact confirmed class index")
        if record["class_key"] != classes[class_id]["class_key"]:
            raise ValueError("class_key mismatch; label inference is forbidden")
        return {"kind": "exact", "model_class_ids": [class_id], "group_key": f"class:{class_id}"}
    expected = {
        "kind", "species_catalog_entry_id", "scientific_name", "chinese_name",
        "model_class_ids", "class_keys",
    }
    if set(value) != expected or value["kind"] != "species_group":
        raise ValueError("invalid species-group supervision structure")
    for key in ("species_catalog_entry_id", "scientific_name", "chinese_name"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"invalid species-group field: {key}")
    ids = value["model_class_ids"]
    if (not isinstance(ids, list) or not ids or ids != sorted(set(ids))
            or any(type(i) is not int or not 0 <= i < len(classes) for i in ids)):
        raise ValueError("invalid species-group model_class_ids")
    if value["class_keys"] != [classes[i]["class_key"] for i in ids]:
        raise ValueError("species-group class_keys do not exactly match the class table")
    if record.get("model_class_id") is not None or record.get("class_key") is not None:
        raise ValueError("coarse species supervision cannot claim one exact output class")
    return {"kind": "species_group", "model_class_ids": ids,
            "group_key": "species:" + value["species_catalog_entry_id"]}


def validate_parent(record, classes, provenance, *, excluded=False):
    require_split(record["split"])
    for key in ("record_id", "split_group_id", "storage_path", "source_dataset"):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f"missing parent identity: {key}")
    target = supervision(record, classes)
    require_hash(record["sha256"])
    for key in ("width", "height", "byte_size"):
        if type(record[key]) is not int or record[key] <= 0:
            raise ValueError(f"invalid {key}")
    refs = record["review_source_ids"]
    if not isinstance(refs, list) or not refs or len(set(refs)) != len(refs):
        raise ValueError("missing/duplicate review sources")
    if any(ref not in provenance for ref in refs):
        raise ValueError("unknown review source")
    instances = record["instances"]
    if not isinstance(instances, list):
        raise ValueError("instances must be an explicit list")
    if excluded:
        if instances or record["exclusion_reason"] != "final_zero_boxes":
            raise ValueError("zero-box exclusion must have no instances")
    elif not instances:
        raise ValueError("final zero-box parents cannot enter supervised Dataset")
    seen = set()
    boxes = set()
    for instance in instances:
        instance_id = instance["instance_id"]
        if not isinstance(instance_id, str) or not instance_id or instance_id in seen:
            raise ValueError("invalid/duplicate instance_id")
        seen.add(instance_id)
        if target["kind"] == "exact":
            if (type(instance["model_class_id"]) is not int
                    or instance["model_class_id"] != record["model_class_id"]
                    or instance["class_key"] != record["class_key"]
                    or instance["exact_class_confirmed"] is not True):
                raise ValueError("instance exact class must be confirmed against parent")
        elif (instance.get("model_class_id") is not None
              or instance.get("class_key") is not None
              or instance.get("exact_class_confirmed") is not False
              or instance.get("supervision") != record["supervision"]):
            raise ValueError("coarse instance supervision must exactly match its parent")
        if instance["origin"] not in ORIGINS:
            raise ValueError("unknown box origin")
        if instance["review_source_id"] not in refs:
            raise ValueError("instance missing bound review source")
        bbox = normalized_box(instance["bbox_xyxy_normalized"])
        if tuple(bbox) in boxes:
            raise ValueError("duplicate final bbox")
        boxes.add(tuple(bbox))
        expanded, pixels = crop_geometry(bbox, record["width"], record["height"])
        if (instance["crop_expansion_fraction_per_side"] != 0.15
                or normalized_box(instance["expanded_bbox_xyxy_normalized"]) != expanded
                or instance["crop_bbox_xyxy_pixels"] != pixels
                or any(type(x) is not int for x in instance["crop_bbox_xyxy_pixels"])):
            raise ValueError("frozen crop geometry/rule mismatch")


@dataclass
class ViewBundle:
    root: Path
    records: list
    exclusions: list
    classes: list
    contract: dict
    contract_sha256: str


def load_bundle(root: Path, contract_path: Path, expected_sha256: str, *, synthetic=False):
    contract = json.loads(verified_bytes(contract_path, expected_sha256))
    if contract["format"] != FORMAT:
        raise ValueError("not a dual-view classification contract")
    # Check the split declaration BEFORE opening ANY referenced manifest.
    if set(contract["manifests"]) != set(ALLOWED_SPLITS):
        raise ValueError("contract must contain only train/validation; final_test is sealed")
    required_status = "synthetic" if synthetic else "frozen_user_confirmed"
    if contract["status"] != required_status:
        raise ValueError(f"data contract status must be {required_status}")
    if contract["crop_rule"] != CROP_RULE or contract["zero_box_rule"] != ZERO_RULE:
        raise ValueError("classification crop/zero-box contract mismatch")
    def read_ref(ref):
        return verified_bytes(relative_file(root, ref["storage_path"]), ref["sha256"])
    # Class table is parsed from the exact same bytes whose digest was checked.
    class_data = read_ref(contract["class_map"])
    classes = json.loads(class_data)["classes"]
    if (not classes or [c["model_class_id"] for c in classes] != list(range(len(classes)))
            or any(type(c["model_class_id"]) is not int for c in classes)
            or any(not isinstance(c["class_key"], str) or not c["class_key"] for c in classes)
            or len({c["class_key"] for c in classes}) != len(classes)):
        raise ValueError("invalid exact class table")
    hierarchy = contract.get("class_hierarchy")
    if hierarchy is not None:
        expected = {
            "format", "same_species_top1_credit",
            "coarse_species_correct_if_prediction_in_group", "groups",
            "class_to_species_group",
        }
        if (set(hierarchy) != expected
                or hierarchy["format"] != "birdsvision-class-output-species-hierarchy-v1"
                or hierarchy["same_species_top1_credit"] != 0.85
                or hierarchy["coarse_species_correct_if_prediction_in_group"] is not True
                or len(hierarchy["class_to_species_group"]) != len(classes)):
            raise ValueError("invalid class hierarchy contract")
        seen_groups = set()
        for group in hierarchy["groups"]:
            probe = {"model_class_id": None, "class_key": None,
                     "supervision": {"kind": "species_group", **group}}
            target = supervision(probe, classes)
            if target["group_key"] in seen_groups:
                raise ValueError("duplicate species group in class hierarchy")
            seen_groups.add(target["group_key"])
            for class_id in target["model_class_ids"]:
                if hierarchy["class_to_species_group"][class_id] != group["species_catalog_entry_id"]:
                    raise ValueError("class hierarchy reverse mapping mismatch")
    provenance = contract["review_sources"]
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("missing review source contract")
    for ref in provenance.values():
        for key in ("source_format", "evidence_kind"):
            if not isinstance(ref[key], str) or not ref[key]:
                raise ValueError("missing review provenance description")
        read_ref(ref)
    if "classification_view_overrides" in contract:
        read_ref(contract["classification_view_overrides"])
    if "unconfirmed_parent_omissions" in contract:
        read_ref(contract["unconfirmed_parent_omissions"])
    records, exclusions = [], []
    seen, seen_instances, groups, images = set(), set(), {}, {}
    def register(row):
        if row["record_id"] in seen:
            raise ValueError("duplicate parent record_id across manifests/exclusions")
        seen.add(row["record_id"])
        for instance in row["instances"]:
            if instance["instance_id"] in seen_instances:
                raise ValueError("instance_id reused across parents")
            seen_instances.add(instance["instance_id"])
        for value, index in ((row["split_group_id"], groups), (row["sha256"], images)):
            if value in index and index[value] != row["split"]:
                raise ValueError("cross-split group/image leakage")
            index[value] = row["split"]
        relative_file(root, row["storage_path"])
    for split in ALLOWED_SPLITS:
        rows = [json.loads(line) for line in read_ref(contract["manifests"][split]).splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"empty {split} manifest")
        for row in rows:
            require_split(row["split"])
            if row["split"] != split:
                raise ValueError("manifest split mismatch")
            validate_parent(row, classes, provenance)
            register(row)
            records.append(row)
    for line in read_ref(contract["exclusions"]).splitlines():
        if line.strip():
            row = json.loads(line)
            validate_parent(row, classes, provenance, excluded=True)
            register(row)
            exclusions.append(row)
    summary = {
        "parents_by_split": dict(sorted(Counter(r["split"] for r in records).items())),
        "split_groups": len(groups),
        "parent_group_mapping_sha256": digest(canonical([
            [r["record_id"], r["split"], r["split_group_id"]] for r in records + exclusions
        ])),
        "zero_box_exclusions": len(exclusions),
        "zero_box_by_source_split_class": dict(sorted(Counter(
            json.dumps([r["source_dataset"], r["split"], r["model_class_id"]]) for r in exclusions
        ).items())),
    }
    if contract["parent_summary"] != summary:
        raise ValueError("parent grouping/exclusion summary mismatch")
    return ViewBundle(root.resolve(), records, exclusions, classes, contract, expected_sha256)


class DualViewDataset(Dataset):
    def __init__(self, bundle: ViewBundle, split: str, transform, *, seed=0, epoch=0):
        require_split(split)
        self.root = bundle.root
        self.records = copy.deepcopy(bundle.records)
        for record in self.records:
            validate_parent(record, bundle.classes, bundle.contract["review_sources"])
        self.records = [r for r in self.records if r["split"] == split]
        if not self.records:
            raise ValueError("empty supervised Dataset")
        self.transform, self.seed, self.epoch = transform, seed, epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        data = verified_bytes(relative_file(self.root, record["storage_path"]),
                              record["sha256"], record["byte_size"])
        with Image.open(io.BytesIO(data)) as source:
            if source.size != (record["width"], record["height"]):
                raise ValueError("image dimensions mismatch")
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("animated source requires a separately frozen static derivative")
            image = source.convert("RGB")
            image.load()
        views = [image] + [image.crop(i["crop_bbox_xyxy_pixels"]) for i in record["instances"]]
        # Per-parent randomness is independent of worker scheduling and resumed
        # batch cursor. Do not perturb model/dropout or caller random state.
        sample_seed = int(digest(canonical([self.seed, self.epoch, record["record_id"]]))[:16], 16)
        random_state = random.getstate()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(sample_seed)
                random.seed(sample_seed)
                tensors = [self.transform(view) for view in views]
        finally:
            random.setstate(random_state)
        return {"views": tensors, "record": record}


def collate_views(samples, *, max_views):
    if not samples or sum(len(s["views"]) for s in samples) > max_views:
        raise ValueError("empty batch or max_views exceeded; parents cannot be truncated")
    views, parents, roles, instance_ids = [], [], [], []
    for parent, sample in enumerate(samples):
        instances = sample["record"]["instances"]
        if len(sample["views"]) != 1 + len(instances):
            raise ValueError("view/instance mapping mismatch")
        views.extend(sample["views"])
        parents.extend([parent] * len(sample["views"]))
        roles.extend([0] + [1] * len(instances))
        instance_ids.extend([None] + [i["instance_id"] for i in instances])
    flat = torch.stack(views)
    if flat.ndim != 4 or flat.shape[1] != 3:
        raise ValueError("flat_views must be [V, 3, H, W]")
    return {
        "flat_views": flat, "view_parent": torch.tensor(parents, dtype=torch.long),
        "view_role": torch.tensor(roles, dtype=torch.long), "instance_ids": instance_ids,
        "labels": torch.tensor([
            s["record"]["model_class_id"] if s["record"].get("supervision") is None else -1
            for s in samples
        ]),
        "target_class_ids": [
            [s["record"]["model_class_id"]] if s["record"].get("supervision") is None
            else list(s["record"]["supervision"]["model_class_ids"])
            for s in samples
        ],
        "records": [s["record"] for s in samples],
    }


class ViewBudgetBatchSampler(Sampler):
    """Greedy batches of WHOLE parents; oversized parents fail, never drop crops."""
    def __init__(self, records, max_views, *, shuffle=False, seed=0):
        if type(max_views) is not int or max_views <= 0:
            raise ValueError("max_views must be a positive integer")
        self.counts = [1 + len(r["instances"]) for r in records]
        if not self.counts or max(self.counts) > max_views:
            raise ValueError("a whole parent exceeds max_views")
        self.max_views, self.shuffle, self.seed = max_views, shuffle, seed

    def __iter__(self):
        indices = list(range(len(self.counts)))
        if self.shuffle:
            random.Random(self.seed).shuffle(indices)
        batch, used = [], 0
        for index in indices:
            count = self.counts[index]
            if used + count > self.max_views:
                yield batch
                batch, used = [], 0
            batch.append(index)
            used += count
        if batch:
            yield batch

    def __len__(self):
        return sum(1 for _ in self)


def synthetic_bundle(root: Path, *, num_classes=1224):
    """Create only temporary artificial data; caller must supply a NEW directory.

    Fields for exact class confirmation and review_source_ids are new export
    fields. Neither locator review format alone supplies this final contract.
    """
    root.mkdir(parents=True, exist_ok=False)
    def write(name, value, jsonl=False):
        data = b"".join(canonical(r) + b"\n" for r in value) if jsonl else canonical(value)
        (root / name).write_bytes(data)
        return {"storage_path": name, "sha256": digest(data)}
    classes = [{"model_class_id": i, "class_key": f"synthetic:{i}"} for i in range(num_classes)]
    class_ref = write("classes.json", {"classes": classes})
    evidence = write("synthetic_review.json", {"purpose": "constructed test evidence only"})
    evidence.update(source_format="synthetic-v1", evidence_kind="synthetic_confirmation")
    rows = []
    for index, (split, count) in enumerate((("train", 1), ("train", 2), ("validation", 1), ("train", 0))):
        path = root / f"image-{index}.png"
        Image.new("RGB", (80, 60), (30 + index * 40, 90, 150)).save(path)
        record = {"record_id": f"synthetic:{index}", "model_class_id": index % num_classes,
                  "class_key": classes[index % num_classes]["class_key"], "split": split,
                  "split_group_id": f"synthetic-group:{index}", "storage_path": path.name,
                  "sha256": digest(path.read_bytes()), "byte_size": path.stat().st_size,
                  "width": 80, "height": 60, "source_dataset": "synthetic",
                  "review_source_ids": ["synthetic-review"], "instances": []}
        for n in range(count):
            bbox = [0.1 + n * 0.35, 0.2, 0.4 + n * 0.35, 0.8]
            expanded, pixels = crop_geometry(bbox, 80, 60)
            record["instances"].append({
                "instance_id": f"synthetic:{index}#instance:{n}",
                "model_class_id": record["model_class_id"], "class_key": record["class_key"],
                "exact_class_confirmed": True, "origin": "manual_box",
                "review_source_id": "synthetic-review", "bbox_xyxy_normalized": bbox,
                "crop_expansion_fraction_per_side": 0.15,
                "expanded_bbox_xyxy_normalized": expanded, "crop_bbox_xyxy_pixels": pixels,
            })
        rows.append(record)
    rows[-1]["exclusion_reason"] = "final_zero_boxes"
    contract = {"format": FORMAT, "status": "synthetic", "crop_rule": CROP_RULE,
                "zero_box_rule": ZERO_RULE, "class_map": class_ref,
                "review_sources": {"synthetic-review": evidence},
                "manifests": {split: write(f"{split}.jsonl", [r for r in rows[:-1] if r["split"] == split], True)
                              for split in ALLOWED_SPLITS},
                "exclusions": write("exclusions.jsonl", rows[-1:], True),
                "parent_summary": {
                    "parents_by_split": {"train": 2, "validation": 1}, "split_groups": 4,
                    "parent_group_mapping_sha256": digest(canonical([
                        [r["record_id"], r["split"], r["split_group_id"]] for r in rows])),
                    "zero_box_exclusions": 1,
                    "zero_box_by_source_split_class": {json.dumps(["synthetic", "train", rows[-1]["model_class_id"]]): 1},
                }}
    ref = write("data_contract.json", contract)
    return load_bundle(root, root / ref["storage_path"], ref["sha256"], synthetic=True)
