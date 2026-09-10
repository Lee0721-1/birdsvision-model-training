# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
import random
import sys
from pathlib import Path

import pytest
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dual_view_finetune_training_data as fine_data
import dual_view_training_data as base_data


def to_tensor(_image):
    return torch.zeros(3, 8, 8)


class TaggedTransform:
    def __init__(self, value, calls):
        self.value = value
        self.calls = calls

    def __call__(self, image):
        self.calls.append(image.size)
        return torch.full((3, 8, 8), self.value)


def test_tiny_train_crop_uses_gentle_transform_without_changing_full_or_normal_crop(tmp_path):
    bundle = base_data.synthetic_bundle(tmp_path / "data", num_classes=4)
    record = bundle.records[1]
    instance = record["instances"][0]
    instance["bbox_xyxy_normalized"] = [0.1, 0.1, 0.15, 0.15]
    expanded, pixels = base_data.crop_geometry(
        instance["bbox_xyxy_normalized"], record["width"], record["height"]
    )
    instance["expanded_bbox_xyxy_normalized"] = expanded
    instance["crop_bbox_xyxy_pixels"] = pixels
    calls = {"full": [], "crop": [], "tiny": []}
    dataset = fine_data.FineTuneDualViewDataset(
        bundle, "train", TaggedTransform(1, calls["full"]),
        TaggedTransform(2, calls["crop"]), TaggedTransform(3, calls["tiny"]),
        tiny_crop_area_threshold=0.01,
    )
    sample = dataset[1]
    assert [view[0, 0, 0].item() for view in sample["views"]] == [1, 3, 2]
    assert [len(calls[name]) for name in ("full", "crop", "tiny")] == [1, 1, 1]


def test_validation_never_uses_train_only_tiny_transform(tmp_path):
    bundle = base_data.synthetic_bundle(tmp_path / "data", num_classes=4)
    calls = {"full": [], "crop": [], "tiny": []}
    dataset = fine_data.FineTuneDualViewDataset(
        bundle, "validation", TaggedTransform(1, calls["full"]),
        TaggedTransform(2, calls["crop"]), None,
        tiny_crop_area_threshold=0.01,
    )
    sample = dataset[0]
    assert [view[0, 0, 0].item() for view in sample["views"]] == [1, 2]
    assert len(calls["tiny"]) == 0


def test_role_aware_augmentation_is_repeatable_and_preserves_callers_rng(tmp_path):
    bundle = base_data.synthetic_bundle(tmp_path / "data", num_classes=4)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(16), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = fine_data.FineTuneDualViewDataset(
        bundle, "train", transform, transform, transform,
        tiny_crop_area_threshold=0.01, seed=9, epoch=3,
    )
    torch_state = torch.get_rng_state().clone()
    python_state = random.getstate()
    first = dataset[0]
    second = dataset[0]
    assert torch.equal(torch_state, torch.get_rng_state())
    assert python_state == random.getstate()
    assert all(torch.equal(a, b) for a, b in zip(first["views"], second["views"], strict=True))


@pytest.mark.parametrize("value", [0, 1, -0.1, float("nan"), True])
def test_tiny_threshold_is_strict(value, tmp_path):
    bundle = base_data.synthetic_bundle(tmp_path / "data", num_classes=4)
    with pytest.raises(ValueError, match="tiny_crop_area_threshold"):
        fine_data.FineTuneDualViewDataset(
            bundle, "train", to_tensor, to_tensor, to_tensor,
            tiny_crop_area_threshold=value,
        )
