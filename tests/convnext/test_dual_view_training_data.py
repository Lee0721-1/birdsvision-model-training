# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from convnext import dual_view_training_data as data


@pytest.fixture
def bundle(tmp_path):
    return data.synthetic_bundle(tmp_path / "data", num_classes=4)


def transform():
    return transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])


def test_crop_expands_each_side_then_clips_without_shifting_opposite_edge():
    expanded, pixels = data.crop_geometry([0.0, 0.0, 0.5, 0.5], 100, 80)
    assert expanded == [0.0, 0.0, 0.575, 0.575]
    assert pixels == [0, 0, 58, 46]


def rewrite_contract(bundle, mutate):
    value = copy.deepcopy(bundle.contract)
    mutate(value)
    path = bundle.root / "changed_contract.json"
    path.write_bytes(data.canonical(value))
    return path, data.digest(path.read_bytes())


def test_single_multiple_mapping_and_source_bytes_unchanged(bundle):
    before = {p.name: p.read_bytes() for p in bundle.root.glob("*.png")}
    dataset = data.DualViewDataset(bundle, "train", transform())
    single, multiple = dataset[0], dataset[1]
    assert len(single["views"]) == 2
    assert len(multiple["views"]) == 3
    batch = data.collate_views([single, multiple], max_views=5)
    assert batch["flat_views"].shape == (5, 3, 32, 32)
    assert batch["view_parent"].tolist() == [0, 0, 1, 1, 1]
    assert batch["view_role"].tolist() == [0, 1, 0, 1, 1]
    assert batch["labels"].tolist() == [0, 1]
    assert batch["instance_ids"] == [None, "synthetic:0#instance:0", None,
                                      "synthetic:1#instance:0", "synthetic:1#instance:1"]
    for parent, record in enumerate(batch["records"]):
        for i in record["instances"]:
            assert i["model_class_id"] == batch["labels"][parent]
        assert record["split"] == "train"
        assert record["split_group_id"] == f"synthetic-group:{parent}"
    assert before == {p.name: p.read_bytes() for p in bundle.root.glob("*.png")}


def test_zero_boxes_are_audited_and_never_loaded(bundle, monkeypatch):
    excluded = bundle.exclusions[0]
    assert excluded["model_class_id"] == 3
    assert excluded["instances"] == []
    assert bundle.contract["parent_summary"]["zero_box_exclusions"] == 1
    bundle.records.append(excluded)
    monkeypatch.setattr(data, "verified_bytes", lambda *a, **kw: pytest.fail("must reject before image read"))
    with pytest.raises(ValueError, match="zero-box"):
        data.DualViewDataset(bundle, "train", transform())


@pytest.mark.parametrize("box", [
    [-0.1, 0.2, 0.8, 0.9], [0.1, 0.2, 1.1, 0.9], [0.1, 0.2, 0.1, 0.9],
    [0.8, 0.2, 0.1, 0.9], [float("nan"), 0.2, 0.8, 0.9],
    [float("inf"), 0.2, 0.8, 0.9], [True, 0.2, 0.8, 0.9],
])
def test_bad_geometry_rejected(bundle, box):
    bundle.records[0]["instances"][0]["bbox_xyxy_normalized"] = box
    with pytest.raises(ValueError, match="bbox"):
        data.DualViewDataset(bundle, "train", transform())


@pytest.mark.parametrize("field,value", [
    ("crop_expansion_fraction_per_side", 0.1),
    ("expanded_bbox_xyxy_normalized", [0.0, 0.0, 1.0, 1.0]),
    ("crop_bbox_xyxy_pixels", [0, 0, 80, 60]),
    ("exact_class_confirmed", False), ("model_class_id", None),
    ("class_key", "SYNTHETIC:0"), ("review_source_id", "unbound"),
])
def test_crop_label_and_provenance_are_exact(bundle, field, value):
    bundle.records[0]["instances"][0][field] = value
    with pytest.raises(ValueError):
        data.DualViewDataset(bundle, "train", transform())


def test_missing_review_sources_and_duplicate_boxes_rejected(bundle):
    original = copy.deepcopy(bundle.records)
    bundle.records[0]["review_source_ids"] = []
    with pytest.raises(ValueError, match="review sources"):
        data.DualViewDataset(bundle, "train", transform())
    bundle.records = original
    duplicate = copy.deepcopy(bundle.records[0]["instances"][0])
    duplicate["instance_id"] = "different-id-same-box"
    bundle.records[0]["instances"].append(duplicate)
    with pytest.raises(ValueError, match="duplicate final bbox"):
        data.DualViewDataset(bundle, "train", transform())


def test_source_hash_size_and_dimensions_rechecked_on_access(bundle):
    dataset = data.DualViewDataset(bundle, "train", transform())
    record = dataset.records[0]
    path = bundle.root / record["storage_path"]
    raw = path.read_bytes()
    path.write_bytes(raw + b"x")
    with pytest.raises(ValueError, match="byte_size"):
        dataset[0]
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ValueError, match="SHA-256"):
        dataset[0]
    path.write_bytes(raw)
    record["width"] += 1
    with pytest.raises(ValueError, match="dimensions"):
        dataset[0]


@pytest.mark.parametrize("path", ["../image.png", "/image.png", "C:/image.png", "a\\b.png", "a//b.png", "./image.png"])
def test_portable_paths_reject_escapes_without_repair(tmp_path, path):
    with pytest.raises(ValueError):
        data.relative_file(tmp_path, path)


def test_source_symlink_rejected(tmp_path):
    source = tmp_path / "real.png"
    source.write_bytes(b"test")
    link = tmp_path / "link.png"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("host does not permit creating symlinks")
    with pytest.raises(ValueError, match="symlinks"):
        data.relative_file(tmp_path, "link.png")


def test_final_test_contract_rejected_before_manifest_open(bundle, monkeypatch):
    path, sha = rewrite_contract(bundle, lambda c: c["manifests"].update(
        final_test={"storage_path": "never-read.jsonl", "sha256": "a" * 64}))
    original = data.verified_bytes
    reads = []
    def spy(path, *args, **kwargs):
        reads.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(data, "verified_bytes", spy)
    with pytest.raises(ValueError, match="final_test"):
        data.load_bundle(bundle.root, path, sha, synthetic=True)
    assert reads == ["changed_contract.json"]
    with pytest.raises(ValueError, match="final_test"):
        data.DualViewDataset(bundle, "final_test", transform())


def test_final_test_row_rejected_before_image_read(bundle):
    bundle.records[0]["split"] = "final_test"
    bundle.records[0]["storage_path"] = "never-read.png"
    with pytest.raises(ValueError, match="final_test"):
        data.DualViewDataset(bundle, "train", transform())


@pytest.mark.parametrize("target", ["manifests", "class_map", "review_sources", "exclusions"])
def test_all_contract_artifact_hashes_verified(bundle, target):
    value = bundle.contract[target]
    ref = value["train"] if target == "manifests" else value["synthetic-review"] if target == "review_sources" else value
    (bundle.root / ref["storage_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        data.load_bundle(bundle.root, bundle.root / "data_contract.json", bundle.contract_sha256, synthetic=True)


@pytest.mark.parametrize("field", ["split_group_id", "sha256"])
def test_cross_split_leakage_rejected(bundle, field):
    validation_path = bundle.root / "validation.jsonl"
    row = json.loads(validation_path.read_text())
    row[field] = bundle.records[0][field]
    validation_path.write_bytes(data.canonical(row) + b"\n")
    path, sha = rewrite_contract(bundle, lambda c: c["manifests"]["validation"].update(
        sha256=data.digest(validation_path.read_bytes())))
    with pytest.raises(ValueError, match="cross-split"):
        data.load_bundle(bundle.root, path, sha, synthetic=True)


def test_view_budget_never_truncates_or_reweights_parents(bundle):
    rows = bundle.records[:2]
    assert list(data.ViewBudgetBatchSampler(rows, 5)) == [[0, 1]]
    assert list(data.ViewBudgetBatchSampler(rows, 4)) == [[0], [1]]
    with pytest.raises(ValueError, match="whole parent"):
        data.ViewBudgetBatchSampler(rows, 2)
    dataset = data.DualViewDataset(bundle, "train", transform())
    with pytest.raises(ValueError, match="max_views"):
        data.collate_views([dataset[0], dataset[1]], max_views=4)
    assert list(data.ViewBudgetBatchSampler(rows, 5, shuffle=True, seed=4)) == list(
        data.ViewBudgetBatchSampler(rows, 5, shuffle=True, seed=4))


def test_current_review_formats_are_not_mistaken_for_classification_contracts(bundle):
    # Exact shapes read from the two current review tools. Supplement box review
    # does not assign a model_class_id or define a frozen split.
    main_state = {"manual_review_format": "birdsvision-china-common-locator-parent-review-v2",
                  "items": {"item": {"parent_review_status": "reviewed",
                                      "review": {"final_instances": []}}}}
    supplement_state = {"manual_review_format": "birdsvision-china-common-high-quality-supplement-locator-parent-review-v1",
                        "items": {"item": {"review_status": "reviewed", "approval_mode": "original_model_boxes",
                                            "approved_original_instance_ids": ["box"], "manual_boxes": {}}}}
    for state in (main_state, supplement_state):
        path = bundle.root / "live-state.json"
        path.write_bytes(data.canonical(state))
        with pytest.raises((ValueError, KeyError)):
            data.load_bundle(bundle.root, path, data.digest(path.read_bytes()))
    bundle.records[0]["model_class_id"] = None
    with pytest.raises(ValueError, match="confirmed"):
        data.DualViewDataset(bundle, "train", transform())


def test_synthetic_status_cannot_start_formal_training(bundle):
    with pytest.raises(ValueError, match="frozen_user_confirmed"):
        data.load_bundle(bundle.root, bundle.root / "data_contract.json", bundle.contract_sha256)


def test_random_transforms_repeat_without_changing_model_rng(bundle):
    augmented = transforms.Compose([transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    dataset = data.DualViewDataset(bundle, "train", augmented, seed=7, epoch=2)
    rng = torch.get_rng_state().clone()
    first, second = dataset[0], dataset[0]
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(first["views"], second["views"], strict=True))
