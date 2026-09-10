# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIVATE = HERE / "private"

DUAL_VIEW_TRAINING_RUN = {
    "allow_training": False,
    "project_root": PRIVATE / "data",
    "data_contract": PRIVATE / "data" / "data_contract.json",
    "data_contract_sha256": "REPLACE_WITH_64_HEX_CHARACTERS",
    "output_dir": PRIVATE / "runs" / "baseline",
    "device": "cuda", "max_views": 64, "epochs": 20, "num_workers": 4,
    "learning_rate": 1e-4, "weight_decay": 0.05, "seed": 20260907,
    "alpha": 0.5, "amp": True, "patience": 5, "max_runtime_minutes": 0,
    "resume": "auto", "init_checkpoint": None,
    "init_checkpoint_sha256": None, "init_class_map": None,
    "init_class_map_sha256": None,
}
