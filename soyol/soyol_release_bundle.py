# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Build a private SOYOL release staging bundle from verified local evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COPIED_REPO_FILES = {
    "MODEL_CARD.md": "SOYOL_MODEL_CARD.md",
    "LICENSE": "LICENSE",
    "THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.md",
}
EXPECTED_ATTRIBUTION_FIELDS = {
    "record_id", "photo_id", "observation_id", "source_page_url",
    "source_image_sha256", "license_code", "license_url",
    "license_url_basis", "attribution", "changes",
    "publication_review", "metadata_checked_at_utc",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checked_source_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    if status.strip():
        raise ValueError("source repository must be clean before staging")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("invalid source repository commit")
    return commit


def validate_inputs(record_path: Path, weight_path: Path,
                    attribution_path: Path, validation_path: Path) -> tuple[dict, dict, int]:
    record = read_json(record_path)
    validation = read_json(validation_path)
    if (record["format"] != "birdsvision-soyol-documented-train-validation-record-v1"
            or record["status"] != "training_and_validation_completed_publication_pending"
            or record["selected_checkpoint"] != "best.pt"
            or record["final_test_read_or_written"] is not False):
        raise ValueError("training record does not describe the selected private release")
    if digest(weight_path) != record["weight_sha256"]["best.pt"]:
        raise ValueError("selected weight does not match the training record")
    if digest(attribution_path) != record["attribution_sha256"]:
        raise ValueError("attribution does not match the training record")
    if digest(validation_path) != record["validation_report_sha256"]:
        raise ValueError("validation report does not match the training record")
    if (validation["format"] != "birdsvision-soyol-a-nms-validation-v1"
            or validation["postprocess"] != "one_to_many_branch_then_nms"
            or validation["max_det"] != 10
            or validation["selected_checkpoint"] != "best"
            or validation["final_test_read_or_written"] is not False
            or validation["dataset_verification"]["status"] != "verified"
            or validation["dataset_verification"]["counts"] != record["validation_split"]):
        raise ValueError("validation report contradicts the training record")
    for name in ("best", "last"):
        metrics = {key: value for key, value in validation["results"][name].items()
                   if key != "weight_path"}
        if metrics != record["nms_validation_metrics"][name]:
            raise ValueError(f"{name} metrics contradict the training record")
    with attribution_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != EXPECTED_ATTRIBUTION_FIELDS:
            raise ValueError("attribution columns do not match the reviewed schema")
        rows = list(reader)
    expected_rows = (record["validation_split"]["train_parents"]
                     + record["validation_split"]["validation_parents"])
    if len(rows) != expected_rows or len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("attribution count or record IDs are inconsistent")
    if any(row["license_code"] not in {"cc-by", "cc0"}
           or not row["attribution"] or not row["source_page_url"].startswith(
               "https://www.inaturalist.org/observations/"
           ) or row["publication_review"] != "platform_terms_reply_pending"
           for row in rows):
        raise ValueError("attribution contains an unreviewed source or license")
    return record, validation, len(rows)


def build_bundle(record_path: Path, weight_path: Path, attribution_path: Path,
                 validation_path: Path, output: Path) -> dict:
    record, validation, attribution_count = validate_inputs(
        record_path, weight_path, attribution_path, validation_path
    )
    output = output.resolve()
    if output == REPO_ROOT or REPO_ROOT in output.parents:
        raise ValueError("weight bundle must stay outside the Git repository")
    if output.exists():
        raise FileExistsError(output)
    source_commit = checked_source_commit()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="soyol-release-", dir=output.parent) as temp:
        bundle = Path(temp)
        shutil.copyfile(weight_path, bundle / "best.pt")
        shutil.copyfile(attribution_path, bundle / "ATTRIBUTION.csv")
        for destination, source in COPIED_REPO_FILES.items():
            shutil.copyfile(REPO_ROOT / source, bundle / destination)
        public_validation = {
            key: value for key, value in validation.items() if key != "results"
        }
        public_validation["results"] = {
            name: {key: value for key, value in result.items()
                   if key != "weight_path"}
            for name, result in validation["results"].items()
        }
        (bundle / "VALIDATION.json").write_text(
            json.dumps(public_validation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (bundle / "README.md").write_text(
            "# SOYOL 私密预发布包\n\n"
            "此包用于核对发布材料，尚未获准公开或上传。"
            "其中不包含原图、裁剪图、分类器权重或私有数据清单。\n\n"
            "`best.pt` 是可考证 A 层训练权重；逐图署名见 `ATTRIBUTION.csv`。"
            "验证结果见 `VALIDATION.json`，没有独立 final_test。"
            "`LICENSE` 适用于源代码，不自动重新许可第三方训练图片；"
            "模型权重的最终发布许可及平台条款仍待确认。\n\n"
            f"训练源码提交：{source_commit}\n",
            encoding="utf-8",
        )
        filenames = ("best.pt", "ATTRIBUTION.csv", "MODEL_CARD.md", "LICENSE",
                     "THIRD_PARTY_NOTICES.md", "VALIDATION.json", "README.md")
        manifest = {
            "format": "birdsvision-soyol-private-release-bundle-v1",
            "status": "private_staging_not_publication_clearance",
            "source_repository": "https://github.com/Lee0721-1/birdsvision-model-training",
            "source_commit": source_commit,
            "training_selection_sha256": record["selection_sha256"],
            "training_contract_sha256": record["training_contract_sha256"],
            "attribution_rows": attribution_count,
            "checkpoint": "best.pt",
            "final_test_completed": False,
            "platform_terms_reply_pending": record["platform_terms_reply_pending"],
            "files": {name: digest(bundle / name) for name in filenames},
        }
        (bundle / "RELEASE_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        bundle.replace(output)
    return manifest


def verify_bundle(directory: Path) -> dict:
    manifest = read_json(directory / "RELEASE_MANIFEST.json")
    if manifest["format"] != "birdsvision-soyol-private-release-bundle-v1":
        raise ValueError("invalid release manifest format")
    for filename, expected_digest in manifest["files"].items():
        if Path(filename).name != filename or digest(directory / filename) != expected_digest:
            raise ValueError(f"release file changed: {filename}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_bundle(args.record, args.weight, args.attribution,
                          args.validation, args.output)
    verify_bundle(args.output)
    print(json.dumps({"status": result["status"], "files": list(result["files"]),
                      "attribution_rows": result["attribution_rows"]},
                     ensure_ascii=False))
