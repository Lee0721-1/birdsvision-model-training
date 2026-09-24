# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Shared ConvNeXt-Tiny model and checkpoint utilities."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn
import torchvision.models as models


CHECKPOINT_FORMAT = "birdsvision-training-checkpoint-v1"
CLASSIFIER_WEIGHT_KEY = "backbone.classifier.2.weight"
CLASSIFIER_BIAS_KEY = "backbone.classifier.2.bias"


class StableCUBModel(nn.Module):
    """Production-compatible ConvNeXt-Tiny with stable ``backbone.*`` keys."""

    def __init__(self, model_name="convnext_tiny", num_classes=755, use_pretrained=True):
        super().__init__()
        if model_name != "convnext_tiny":
            raise ValueError("only convnext_tiny is supported")
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, inputs):
        return self.backbone(inputs)


class StopRequest:
    def __init__(self):
        self.signal_number = None

    def request(self, signal_number, _frame):
        self.signal_number = signal_number
        print(
            f"[train] signal {signal_number} received; saving after this batch",
            flush=True,
        )


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def load_torch_file(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(model: nn.Module):
    data_config = timm.data.resolve_model_data_config(model)
    train_transform = timm.data.create_transform(**data_config, is_training=True)
    validation_transform = timm.data.create_transform(**data_config, is_training=False)
    return train_transform, validation_transform, data_config
