# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIVATE = HERE / "private"

DUAL_VIEW_FINE_TUNING_RUN = {
    "allow_training": False,
    "project_root": PRIVATE / "data",
    "data_contract": PRIVATE / "data" / "data_contract.json",
    "data_contract_sha256": "REPLACE_WITH_64_HEX_CHARACTERS",
    "output_dir": PRIVATE / "runs" / "fusion-v2-aux-010",
    "device": "cuda", "max_views": 64, "epochs": 8, "num_workers": 4,
    "learning_rate": 1e-5, "weight_decay": 0.05, "seed": 20260908,
    "default_alpha": 0.4, "many_box_threshold": 3, "many_box_alpha": 0.25,
    "full_aux_loss_weight": 0.10, "crop_aux_loss_weight": 0.10,
    "tiny_crop_area_threshold": 0.01, "tiny_crop_scale_min": 0.5,
    "amp": True, "patience": 3, "max_runtime_minutes": 0,
    "resume": "auto", "init_checkpoint": PRIVATE / "baseline.pth",
    "init_checkpoint_sha256": "REPLACE_WITH_64_HEX_CHARACTERS",
}
