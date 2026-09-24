# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare a source-and-attribution table without copying training images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import urlsplit


ALLOWED_LICENSES = {"cc-by", "cc0", "cc0-1.0"}
FIELDS = ("record_id", "source_dataset", "source_page_url", "attribution",
          "license_code", "changes", "publication_review")
CHANGES = "Bird detection annotation; resized, cropped, and augmented for model training"


def export(selection: Path, output: Path) -> int:
    if output.exists():
        raise FileExistsError(output)
    rows = []
    seen = set()
    for line in selection.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        record_id = row["record_id"]
        source_dataset = row["source_dataset"]
        source_url = row["source_page_url"]
        attribution = row["attribution"]
        license_code = row["license_code"]
        parsed = urlsplit(source_url)
        if (not record_id or record_id in seen or license_code not in ALLOWED_LICENSES
                or not attribution or parsed.scheme != "https" or not parsed.hostname):
            raise ValueError("selection lacks A-tier attribution evidence")
        seen.add(record_id)
        rows.append({
            "record_id": record_id,
            "source_dataset": source_dataset,
            "source_page_url": source_url,
            "attribution": attribution,
            "license_code": license_code,
            "changes": CHANGES,
            "publication_review": (
                "platform_terms_review_required" if source_dataset == "inaturalist" else
                "underlying_photo_rights_review_required" if source_dataset == "huggingface_bird_species_dataset" else
                "source_review_required"
            ),
        })
    if not rows:
        raise ValueError("selection is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.selection, args.output))
