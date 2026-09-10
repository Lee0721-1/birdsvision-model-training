#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Role-aware training views for the independent dual-view fine-tuning run."""

from __future__ import annotations

import copy
import io
import random

import torch
from PIL import Image
from torch.utils.data import Dataset

from dual_view_training_data import (
    canonical,
    digest,
    relative_file,
    require_split,
    validate_parent,
    verified_bytes,
)


def bbox_area(instance):
    x1, y1, x2, y2 = instance["bbox_xyxy_normalized"]
    return (x2 - x1) * (y2 - y1)


class FineTuneDualViewDataset(Dataset):
    """Use a gentler train transform only for crops whose frozen box is tiny."""

    def __init__(self, bundle, split, full_transform, crop_transform,
                 tiny_crop_transform=None, *, tiny_crop_area_threshold=0.01,
                 seed=0, epoch=0):
        require_split(split)
        if (type(tiny_crop_area_threshold) not in (int, float)
                or not 0 < tiny_crop_area_threshold < 1):
            raise ValueError("tiny_crop_area_threshold must be in (0, 1)")
        if split == "train" and tiny_crop_transform is None:
            raise ValueError("train requires an explicit tiny crop transform")
        self.root = bundle.root
        self.records = copy.deepcopy(bundle.records)
        for record in self.records:
            validate_parent(record, bundle.classes, bundle.contract["review_sources"])
        self.records = [record for record in self.records if record["split"] == split]
        if not self.records:
            raise ValueError("empty supervised Dataset")
        self.split = split
        self.full_transform = full_transform
        self.crop_transform = crop_transform
        self.tiny_crop_transform = tiny_crop_transform
        self.tiny_crop_area_threshold = float(tiny_crop_area_threshold)
        self.seed = seed
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        raw = verified_bytes(
            relative_file(self.root, record["storage_path"]),
            record["sha256"],
            record["byte_size"],
        )
        with Image.open(io.BytesIO(raw)) as source:
            if source.size != (record["width"], record["height"]):
                raise ValueError("image dimensions mismatch")
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("animated source requires a separately frozen static derivative")
            image = source.convert("RGB")
            image.load()
        crops = [image.crop(instance["crop_bbox_xyxy_pixels"])
                 for instance in record["instances"]]
        sample_seed = int(digest(canonical([
            self.seed, self.epoch, record["record_id"]
        ]))[:16], 16)
        random_state = random.getstate()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(sample_seed)
                random.seed(sample_seed)
                tensors = [self.full_transform(image)]
                for crop, instance in zip(crops, record["instances"], strict=True):
                    transform = self.crop_transform
                    if (self.split == "train"
                            and bbox_area(instance) < self.tiny_crop_area_threshold):
                        transform = self.tiny_crop_transform
                    tensors.append(transform(crop))
        finally:
            random.setstate(random_state)
        return {"views": tensors, "record": record}
