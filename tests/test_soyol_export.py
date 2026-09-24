# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
import json

import pytest

from soyol_dataset import digest, render_label, split_for, verify
from soyol_export import export


def test_export_requires_reviewed_a_tier_and_separate_splits(tmp_path):
    train_group = next(f"group-{i}" for i in range(100) if split_for(f"group-{i}") == "train")
    val_group = next(f"group-{i}" for i in range(100) if split_for(f"group-{i}") == "validation")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rows = []
    for number, group in enumerate((train_group, val_group)):
        source = source_dir / f"bird-{number}.jpg"
        source.write_bytes(f"synthetic image {number}".encode())
        instance = {"bbox_xyxy_normalized": [0.1, 0.2, 0.6, 0.8],
                    "provenance": "human_manual_box"}
        rows.append({
            "record_id": f"synthetic-{number}", "soyol_group_id": group,
            "source_image_path": str(source), "source_image_sha256": digest(source),
            "source_page_url": f"https://example.org/photo/{number}",
            "source_split_for_audit_only": "train", "source_scope": "existing_frozen_box_supply",
            "attribution": "Example Photographer", "license_code": "cc0",
            "instances": [instance], "detect_label": render_label([instance]),
        })
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    selection = selection_dir / "selection.jsonl"
    selection.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (selection_dir / "report.json").write_text(json.dumps({
        "selection_sha256": digest(selection), "final_test_read_or_written": False,
        "visual_group_rule": "synthetic distinct groups",
    }), encoding="utf-8")
    data = tmp_path / "detect"
    assert export(selection, data)["counts"]["validation"]["boxes"] == 1
    assert verify(data)["status"] == "verified"

    rows[0]["license_code"] = "cc-by-nc"
    selection.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (selection_dir / "report.json").write_text(json.dumps({
        "selection_sha256": digest(selection), "final_test_read_or_written": False,
        "visual_group_rule": "synthetic distinct groups",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="A-tier"):
        export(selection, tmp_path / "rejected")
