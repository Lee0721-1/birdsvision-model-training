# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""SOYOL v1 limits post-NMS predictions to ten boxes, never ground-truth labels."""

from __future__ import annotations

from typing import Any, Mapping


MAX_DETECTIONS_PER_IMAGE = 10


def with_soyol_max_det(arguments: Mapping[str, Any]) -> dict[str, Any]:
    configured = arguments.get("max_det")
    if configured is not None and configured != MAX_DETECTIONS_PER_IMAGE:
        raise ValueError("SOYOL v1 max_det must be 10")
    result = dict(arguments)
    result["max_det"] = MAX_DETECTIONS_PER_IMAGE
    return result
