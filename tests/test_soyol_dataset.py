# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""External SOYOL dataset integrity checks use only synthetic files."""

import json

import pytest

from soyol_dataset import digest, render_label, split_for, verify
from soyol_output_policy import with_soyol_max_det


def test_output_limit_does_not_restrict_ground_truth():
    assert with_soyol_max_det({"imgsz": 640})["max_det"] == 10
    with pytest.raises(ValueError, match="max_det"):
        with_soyol_max_det({"max_det": 11})
    boxes = [{"bbox_xyxy_normalized": [0.1, 0.1, 0.2, 0.2]}] * 11
    assert len(render_label(boxes).splitlines()) == 11


def test_verifier_rejects_changed_label(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    source = tmp_path / "source.jpg"
    source.write_bytes(b"synthetic-image")
    copied = data / "image.jpg"
    copied.write_bytes(source.read_bytes())
    label = data / "label.txt"
    box = {"bbox_xyxy_normalized": [0.1, 0.2, 0.6, 0.8]}
    label.write_text(render_label([box]), encoding="utf-8")
    selection = tmp_path / "selection.jsonl"
    selection.write_text("synthetic selection\n", encoding="utf-8")
    yaml = data / "soyol_detect.yaml"
    yaml.write_text("names: {0: bird}\n", encoding="utf-8")
    split = split_for("synthetic-group")
    row = {
        "record_id": "synthetic-record", "split": split,
        "soyol_group_id": "synthetic-group", "source_scope": "synthetic-reviewed",
        "source_page_url": "https://example.org/photo/1",
        "source_split_for_audit_only": "train", "license_code": "cc0",
        "attribution": "Example Photographer",
        "source_image_path": str(source), "source_image_sha256": digest(source),
        "copied_image_path": "image.jpg", "copied_image_sha256": digest(copied),
        "label_path": "label.txt", "label_sha256": digest(label),
        "instances": [box | {"provenance": "human_manual_box"}],
    }
    manifest = data / "export_manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    counts = {name: {"parents": 0, "boxes": 0, "no_bird": 0}
              for name in ("train", "validation")}
    counts[split]["parents"] = 1
    counts[split]["boxes"] = 1
    contract = {
        "format": "birdsvision-soyol-a-detect-export-v1",
        "final_test_read_or_written": False,
        "manifest_sha256": digest(manifest), "yaml_sha256": digest(yaml),
        "selection_path": str(selection), "selection_sha256": digest(selection),
        "counts": counts,
    }
    (data / "data_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    assert verify(data)["status"] == "verified"
    label.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="label mismatch"):
        verify(data)
