# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
import csv
import json

import pytest

from soyol_attribution import export


def test_attribution_export_keeps_a_tier_and_rejects_b_tier(tmp_path):
    selection = tmp_path / "selection.jsonl"
    output = tmp_path / "attribution.csv"
    row = {
        "record_id": "synthetic-1", "source_page_url": "https://example.org/photo/1",
        "attribution": "Example Photographer", "license_code": "cc-by",
        "source_dataset": "inaturalist",
    }
    selection.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert export(selection, output) == 1
    with output.open(encoding="utf-8", newline="") as stream:
        exported = list(csv.DictReader(stream))[0]
        assert exported["changes"].startswith("Bird detection")
        assert exported["publication_review"] == "platform_terms_review_required"
    row["license_code"] = "cc-by-nc"
    selection.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="A-tier"):
        export(selection, tmp_path / "rejected.csv")
