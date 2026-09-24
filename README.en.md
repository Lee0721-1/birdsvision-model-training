# BirdsVision Model Training

SOYOL stands for Student YOLO. The internal teacher model is called TYLO (Teacher YOLO); it remains closed source and is mainly used to compare the student's results.

Public training code for the BirdsVision shared dual-view ConvNeXt-Tiny classifier. Images, review ledgers, class tables, frozen evaluation data, checkpoints, and trained weights are intentionally excluded.

The [BirdsVision website](https://www.birdsvision.com.cn/) introduces the app, model development progress, privacy information, and download options. This repository provides training source code; neither the website nor this repository distributes training images or production model weights.

See [README.md](README.md) for setup, smoke tests, and the data-contract boundary. Source code is licensed under AGPL-3.0-only.

The SOYOL Detect exporter, training, and NMS validation entries (`soyol_export.py`, `soyol_train.py`, `soyol_validate.py`) accept an external, hash-bound A-tier selection and dataset export. They require independently authorized images and human-confirmed boxes, and never include images, class labels, private manifests, TYLO weights, trained SOYOL weights, or a final-test result in this repository. The ten-box limit applies after NMS, not to ground-truth labels.
