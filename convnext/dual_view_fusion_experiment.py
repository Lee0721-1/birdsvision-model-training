#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
"""Launch one config-bound dual-view fusion-rule control experiment."""

from __future__ import annotations

import sys
from pathlib import Path

from convnext import dual_view_classifier_finetune as train
from convnext.dual_view_training_data import digest


def invocation_identity(config_path):
    config_path = Path(config_path).resolve()
    entrypoint = Path(__file__).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return {
        "entrypoint_name": entrypoint.name,
        "entrypoint_sha256": digest(entrypoint.read_bytes()),
        "active_config_name": config_path.name,
        "active_config_sha256": digest(config_path.read_bytes()),
    }


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if len(raw_argv) != 2 or raw_argv[0] != train.CONFIG_ARGUMENT:
        return train.main(raw_argv)
    identity = invocation_identity(raw_argv[1])
    original_runtime_contract = train.runtime_contract

    def runtime_contract(device):
        value = original_runtime_contract(device)
        value["experiment_invocation"] = identity
        return value

    train.runtime_contract = runtime_contract
    try:
        return train.main(raw_argv)
    finally:
        train.runtime_contract = original_runtime_contract


if __name__ == "__main__":
    raise SystemExit(main())
