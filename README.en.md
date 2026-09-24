# BirdsVision Model Training

Public training code for the BirdsVision shared dual-view ConvNeXt-Tiny classifier. Images, review ledgers, class tables, frozen evaluation data, checkpoints, and trained weights are intentionally excluded.

See [README.md](README.md) for setup, smoke tests, and the data-contract boundary. Source code is licensed under AGPL-3.0-only.

The SOYOL Detect exporter, training, and NMS validation entries (`soyol_export.py`, `soyol_train.py`, `soyol_validate.py`) accept an external, hash-bound A-tier selection and dataset export. They require independently authorized images and human-confirmed boxes, and never include images, class labels, private manifests, TYLO weights, trained SOYOL weights, or a final-test result in this repository. The ten-box limit applies after NMS, not to ground-truth labels.
