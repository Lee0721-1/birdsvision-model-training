# BirdsVision Classifier Training

This repository is not yet publicly accessible. The plan is to publish its ConvNeXt classifier training source. SOYOL localization is in the independent [SOYOL repository](https://github.com/Lee0721-1/birdsvision-soyol-locator), and the classifier API source is in the [inference-server repository](https://github.com/Lee0721-1/birdsvision-inference-server). The Git history still contains SOYOL files from before the split and needs review before publication.

The classifier and locator are maintained as separate projects. Classifier source, labels, and production weights are outside the SOYOL release.

This repository contains source for BirdsVision shared dual-view ConvNeXt-Tiny classifier training. Images, review ledgers, class tables, frozen evaluation data, checkpoints, and trained weights are excluded.

The [BirdsVision website](https://www.birdsvision.com.cn/) introduces the app, model development progress, privacy information, and download options. This repository provides training source code; neither the website nor this repository distributes training images or production model weights.

The `convnext/` directory contains classifier training code and examples; `tests/convnext/` contains its tests. Run module commands from the repository root.

See [README.md](README.md) for setup, smoke tests, and the data-contract boundary. Source code is licensed under AGPL-3.0-only.

Source files retain their AGPL-3.0-only notices. Classifier weights, labels, and training data are excluded from the planned source publication. This repository is not the source release for SOYOL.
