# BirdsVision Model Training

> This is a private historical workspace containing source for both the classifier and the locator. Do not make the entire repository public. The classifier source and weights stay in the private project; SOYOL AGPL release materials are being assembled as a separate locator project.

SOYOL stands for Student YOLO. The internal teacher model is called TYLO (Teacher YOLO); it remains closed source and is mainly used to compare the student's results.

Historical preparation code for the BirdsVision shared dual-view ConvNeXt-Tiny classifier and SOYOL locator. The mixed repository remains private. Images, review ledgers, class tables, frozen evaluation data, checkpoints, and trained weights are excluded.

The [BirdsVision website](https://www.birdsvision.com.cn/) introduces the app, model development progress, privacy information, and download options. This repository provides training source code; neither the website nor this repository distributes training images or production model weights.

The `convnext/` and `soyol/` directories contain their respective training code and examples; `tests/convnext/` and `tests/soyol/` contain their tests. Run module commands from the repository root.

See [README.md](README.md) for setup, smoke tests, and the data-contract boundary. Source code is licensed under AGPL-3.0-only.

The SOYOL Detect exporter, training, and NMS validation entries (`soyol/soyol_export.py`, `soyol/soyol_train.py`, `soyol/soyol_validate.py`) accept an external, hash-bound A-tier selection and dataset export. They require independently authorized images and human-confirmed boxes, and never include images, class labels, private manifests, TYLO weights, trained SOYOL weights, or a final-test result in this repository. The ten-box limit applies after NMS, not to ground-truth labels.
