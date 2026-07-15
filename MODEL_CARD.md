# Model Card

## Model identity

| Field | Value |
|---|---|
| Architecture | YOLOv7 |
| Weight file | `best.pt` |
| Weight download | Pending owner-provided Google Drive release link |
| Training dataset | Not provided in the repository |
| Training date | Not provided |
| Input size | 640 × 640 |
| Intended use | Construction-site helmet and restricted-area safety monitoring |

## Class definitions

The repository does not contain the trained checkpoint or its verified class-name file, so class names must not be inferred from the application title.

Replace this table with the exact order used during training:

| Class ID | Class name | Definition |
|---:|---|---|
| 0 | To be verified | Confirm from the training dataset/checkpoint |
| 1 | To be verified | Confirm from the training dataset/checkpoint |
| 2 | To be verified | Confirm from the training dataset/checkpoint |

The `--class-id` option controls which detected classes can trigger restricted-area events. Verify the class ID before deployment.

## Quantitative evaluation

No evaluation output was supplied with the repository. Record results from a held-out test set; do not use training-set results.

| Metric | Overall | Person | Helmet | No helmet |
|---|---:|---:|---:|---:|
| Precision | Not reported | — | — | — |
| Recall | Not reported | — | — | — |
| mAP@0.5 | Not reported | — | — | — |
| mAP@0.5:0.95 | Not reported | — | — | — |

Also report:

- Number of training, validation and test images
- Class counts for each split
- CPU/GPU hardware
- Average FPS and input resolution
- Day, night, occlusion and small-object failure cases
- Confusion matrix and representative false positives/false negatives

## Limitations

Performance depends on camera placement, lighting, occlusion, worker distance, image resolution and how closely deployment conditions match the training data. This prototype must not be the only construction-site safety control.
