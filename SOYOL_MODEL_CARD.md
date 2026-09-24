# SOYOL v1 model card — preparation record

**Publication status:** Source preparation only. No SOYOL weights, images, labels, or final-test result are included in this private repository.

## Model and intended output

SOYOL v1 is a single-class YOLO26n Detect bird locator. It returns zero to ten post-NMS bird boxes for one image. It does not classify bird species, reject non-bird images, or output keypoints. In the BirdsVision 1.0.2 service it supplies crops for a separate classifier; that classifier's production weights are outside this repository.

## Training provenance currently recorded internally

- Initialization: official Ultralytics `yolo26n.pt` Detect checkpoint, SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`; no TYLO Pose checkpoint was loaded.
- Supervision: A-tier images only, using strict human-confirmed final boxes. The export contains 1,495 images, 1,428 with boxes and 67 explicitly reviewed as having no bird box. Train has 1,220 images and 1,656 boxes; validation has 275 images and 360 boxes. B-tier authorization is deferred; C-tier images were not used for weight updates.
- Run: Ultralytics 8.4.126, 20 complete epochs, image size 640, batch 8. The saved best checkpoint came from epoch 18.
- Validation: for the selected checkpoint on the one-to-many NMS branch, mAP50 was 0.816 and mAP50-95 was 0.515. Inference uses `conf=0.25`, `iou=0.7`, and `max_det=10`. The ten-box limit never truncates ground-truth labels.

These numbers come from the internal training and validation records. The independent SOYOL `final_test` has not been run or accepted. An A-tier image quality review is also recorded as `not_started`. Do not present this preparation record as a completed model release or a guarantee of detection quality.

## Attribution and publication boundary

The internal A-tier selection contains 1,093 CC BY records, 228 CC0 records, and 174 CC0-1.0 records. Each currently has a source page and attribution field. `soyol_attribution.py` prepares a per-record attribution table for review without exporting images. Source page, author attribution, image-level license, and change notices must be checked before publication. The license on this repository's source code does not relicense the training images.

Of these A-tier records, 1,321 were obtained from iNaturalist and 174 from a Hugging Face bird-species dataset. iNaturalist's current platform terms prohibit use of its data to train AI/ML for commercial purposes. The image-level CC BY or CC0 code does not by itself resolve the platform-terms question for a model whose downstream commercial use is intended. This must be resolved through a documented platform permission or independently verified source route before an unrestricted AGPL model-weight release is claimed.

The Hugging Face dataset card marks the dataset CC0-1.0 but says it was sourced from a Kaggle dataset. All 174 selected records currently use the same dataset-level attribution string; the recorded evidence does not identify the original photographer or an image-specific rights grant. These images therefore need upstream, image-level rights verification before the current weights can be described as cleared for public commercial reuse. The attribution exporter marks both source groups as requiring review; its CSV is not a publication approval.

The planned code and model release follows the Ultralytics AGPL-3.0 route. Actual model weights, complete corresponding deployment source, third-party notices, an attribution table, independent acceptance results, and a fixed release manifest must be assembled and reviewed before switching either GitHub repository to Public or creating a weight release.
