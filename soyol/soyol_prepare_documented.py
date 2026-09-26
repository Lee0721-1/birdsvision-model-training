# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Derive a documented iNaturalist-only SOYOL retraining selection.

This keeps the historical selection and split groups intact. It excludes the
five unavailable iNaturalist observations and 174 Hugging Face images whose
original-photo rights chain has not been documented. No images are read here.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from soyol.soyol_dataset import digest, split_for


LICENSE_MAPPING_URL = "https://github.com/inaturalist/inaturalist/blob/main/app/models/shared/license_module.rb"
LICENSE_URLS = {
    "cc-by": "https://creativecommons.org/licenses/by/4.0/",
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
}
ATTRIBUTION_FIELDS = (
    "record_id", "photo_id", "observation_id", "source_page_url",
    "source_image_sha256", "license_code", "license_url",
    "license_url_basis", "attribution", "changes", "publication_review",
    "metadata_checked_at_utc",
)
CHANGES = "Bird detection annotation; resized, cropped, and augmented for model training"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def prepare(selection: Path, attribution_dir: Path, metadata_dir: Path, output: Path) -> dict:
    selection = selection.resolve()
    attribution_dir = attribution_dir.resolve()
    metadata_dir = metadata_dir.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    parent_report = json.loads((selection.parent / "report.json").read_text(encoding="utf-8"))
    if parent_report["format"] != "birdsvision-soyol-a-detect-selection-v1":
        raise ValueError("unexpected parent selection format")
    parent_sha = digest(selection)
    if parent_report["selection_sha256"] != parent_sha:
        raise ValueError("parent selection/report mismatch")
    attribution_summary = json.loads((attribution_dir / "summary.json").read_text(encoding="utf-8"))
    if attribution_summary["selection_sha256"] != parent_sha:
        raise ValueError("attribution overlay is not bound to parent selection")
    metadata_summary = json.loads((metadata_dir / "summary.json").read_text(encoding="utf-8"))
    if metadata_summary["selected_inaturalist_photos"] != 1321:
        raise ValueError("metadata audit size changed")

    parent = list(read_jsonl(selection))
    attribution = {row["record_id"]: row for row in read_jsonl(attribution_dir / "records.jsonl")}
    metadata = {row["record_id"]: row for row in read_jsonl(metadata_dir / "records.jsonl")}
    if len(parent) != 1495 or len({row["record_id"] for row in parent}) != len(parent):
        raise ValueError("parent selection changed")
    if len(attribution) != 1321 or len(metadata) != 1321:
        raise ValueError("photo evidence count changed")

    kept, excluded, attribution_rows = [], [], []
    group_decisions = defaultdict(set)
    for row in parent:
        rid = row["record_id"]
        source = row["source_dataset"]
        reason = None
        if source == "huggingface_bird_species_dataset":
            reason = "upstream_original_photo_rights_chain_unverified"
        elif source == "inaturalist":
            att = attribution[rid]
            meta = metadata[rid]
            if att["source_image_sha256"] != row["source_image_sha256"] or att["source_page_url"] != row["source_page_url"]:
                raise ValueError(f"attribution source mismatch: {rid}")
            if meta["source_page_url"] != row["source_page_url"] or meta["recorded_license_code"] != row["license_code"]:
                raise ValueError(f"metadata source mismatch: {rid}")
            if att["publication_review_status"] == "hold_observation_not_returned":
                if meta["review_issue"] != "observation_not_found" or meta["photo_found"]:
                    raise ValueError(f"hold evidence mismatch: {rid}")
                reason = "original_observation_unavailable"
            elif att["publication_review_status"] in (
                "current_attribution_selected", "current_attribution_selected_over_old_text"
            ):
                if not meta["photo_found"] or not meta["license_code_matches"] or meta["review_issue"] not in (
                    "none", "attribution_changed_or_missing"
                ):
                    raise ValueError(f"photo evidence mismatch: {rid}")
                if att["proposed_attribution_text"] != meta["current_photo_attribution"]:
                    raise ValueError(f"current attribution mismatch: {rid}")
                if row["license_code"] not in LICENSE_URLS or meta["current_photo_license_code"] != row["license_code"]:
                    raise ValueError(f"unmapped or changed photo licence: {rid}")
                updated = dict(row)
                updated["attribution"] = att["proposed_attribution_text"]
                updated["license_url"] = LICENSE_URLS[row["license_code"]]
                updated["license_url_basis"] = "current_site_code_mapping_not_historical_photo_page_capture"
                updated["metadata_checked_at_utc"] = meta["metadata_checked_at_utc"]
                kept.append(updated)
                attribution_rows.append({
                    "record_id": rid, "photo_id": meta["photo_id"],
                    "observation_id": meta["observation_id"],
                    "source_page_url": row["source_page_url"],
                    "source_image_sha256": row["source_image_sha256"],
                    "license_code": row["license_code"],
                    "license_url": updated["license_url"],
                    "license_url_basis": updated["license_url_basis"],
                    "attribution": updated["attribution"],
                    "changes": CHANGES,
                    "publication_review": "platform_terms_reply_pending",
                    "metadata_checked_at_utc": updated["metadata_checked_at_utc"],
                })
            else:
                raise ValueError(f"unexpected publication review status: {rid}")
        else:
            raise ValueError(f"unexpected source: {rid}")
        group_decisions[row["soyol_group_id"]].add("exclude" if reason else "keep")
        if reason:
            excluded.append({
                "record_id": rid, "reason": reason,
                "source_dataset": source,
                "source_image_sha256": row["source_image_sha256"],
                "source_page_url": row["source_page_url"],
                "split_under_parent_policy": split_for(row["soyol_group_id"]),
            })

    if any(len(decisions) != 1 for decisions in group_decisions.values()):
        raise ValueError("visual group crosses keep/exclude boundary")
    if len(kept) != 1316 or len(excluded) != 179:
        raise ValueError("documented selection count changed")
    if Counter(row["reason"] for row in excluded) != Counter({
        "original_observation_unavailable": 5,
        "upstream_original_photo_rights_chain_unverified": 174,
    }):
        raise ValueError("exclusion reasons changed")
    if len({row["record_id"] for row in kept}) != len(kept):
        raise ValueError("duplicate kept record ID")
    if len({row["record_id"] for row in excluded}) != len(excluded):
        raise ValueError("duplicate excluded record ID")

    counts = Counter()
    for row in kept:
        if row["instances"]:
            counts["positive_parents"] += 1
            counts["positive_boxes"] += len(row["instances"])
        else:
            counts["explicit_no_bird_parents"] += 1
    output.mkdir(parents=True)
    selection_path = output / "selection.jsonl"
    with selection_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in kept:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output / "excluded_records.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in excluded:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output / "photo_attribution.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ATTRIBUTION_FIELDS)
        writer.writeheader()
        writer.writerows(attribution_rows)
    report = {
        "format": "birdsvision-soyol-documented-retrain-selection-v1",
        "status": "data_prepared_platform_terms_reply_pending_not_public_release_clearance",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_selection_sha256": parent_sha,
        "source_metadata_sha256": digest(metadata_dir / "records.jsonl"),
        "source_attribution_sha256": digest(attribution_dir / "records.jsonl"),
        "license_mapping_source": LICENSE_MAPPING_URL,
        "license_mapping_basis": "site_source_code_cc_by_4_0_cc0_1_0_not_individual_historical_page_capture",
        "counts": dict(counts),
        "selected_records": len(kept),
        "excluded_records": len(excluded),
        "exclusion_reasons": dict(Counter(row["reason"] for row in excluded)),
        "license_counts": dict(Counter(row["license_code"] for row in kept)),
        "visual_group_rule": parent_report["visual_group_rule"],
        "selection_sha256": digest(selection_path),
        "excluded_records_sha256": digest(output / "excluded_records.jsonl"),
        "photo_attribution_sha256": digest(output / "photo_attribution.csv"),
        "final_test_read_or_written": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--attribution-dir", required=True, type=Path)
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.selection, args.attribution_dir, args.metadata_dir, args.output), ensure_ascii=False, indent=2))
