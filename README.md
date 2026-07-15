# Construction Safety Monitoring

A YOLOv7-based construction-site monitoring project for safety detection and restricted-area intrusion warnings.

## Features

- Video-file and webcam input
- YOLOv7 object detection
- Resolution-independent restricted-area overlay
- Intrusion warning based on the bottom-center point of each detection
- Annotated video export
- Stable tracking IDs with lightweight centroid tracking
- Enter, exit and lost-in-zone event records in CSV format
- Automatic event snapshots
- CPU or CUDA device configuration

## Project structure

- `main.py` — command-line entry point
- `Stream.py` — video capture, inference and output pipeline
- `draw.py` — restricted-area drawing and intrusion checks
- `tracking.py` — tracking IDs and CSV event recording
- `make_demo.py` — generates a GitHub GIF and preview image
- `MODEL_CARD.md` — verified classes and evaluation reporting
- `config.ini` — model and inference settings
- `project_yolov7det/` — YOLOv7 inference implementation

## Model weights

The trained file `best.pt` is not stored in this repository because it is larger than GitHub's 100 MB file limit.

Place the model at:

```text
project_yolov7det/weights/pt/best.pt
```

The expected location can be changed in `config.ini`, or supplied at runtime:

```powershell
python main.py --video "input.mp4" --weights "D:\\models\\best.pt"
```

Large weights and demonstration videos can be stored in Google Drive, but download or sync them locally before execution. The program intentionally does not silently download executable PyTorch weights.

> Only load model files from a source you trust. PyTorch weight files can contain executable serialized content.

### Publishing the weight file

1. Upload `best.pt` to Google Drive.
2. Set sharing to “Anyone with the link — Viewer”.
3. Copy the share link into the **Weight download** row in [MODEL_CARD.md](MODEL_CARD.md).
4. Publish the SHA-256 checksum beside the link so users can verify the file.
5. Keep the file out of Git because it exceeds GitHub's 100 MB limit.

## Installation

Python 3.8–3.10 is recommended. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

PyTorch installation depends on the operating system and CUDA version. If the command above does not install a suitable build, follow the official PyTorch installation selector and then rerun the requirements installation.

## Usage

Calibrate the restricted area for a video or camera first:

```powershell
python calibrate_roi.py --video "path\\to\\input.mp4" --output roi.json
python calibrate_roi.py --camera 0 --output camera-0-roi.json
```

Left-click to add polygon points, right-click to undo, and press Enter to save. Coordinates are stored as normalized values and scale with the video resolution.

Process a video:

```powershell
python main.py --video "path\to\input.mp4" --output output
```

Use webcam 0:

```powershell
python main.py --camera 0 --roi camera-0-roi.json --class-id 0 --output output
```

Process without opening a preview window:

```powershell
python main.py --video "path\to\input.mp4" --no-display
```

Press `q` to stop the preview. Generated files:

- `output/results.avi` — annotated video
- `output/events.csv` — enter, exit and lost-in-zone events
- `output/events/*.jpg` — snapshots captured on entry

Each CSV row includes UTC time, video time, frame number, tracking ID, class ID, confidence, bounding box and snapshot path.

Tracking can be tuned when people move quickly or detections briefly disappear:

```powershell
python main.py --video "input.mp4" --tracker-distance 120 --tracker-missed 30
```

### Generate GitHub demonstration media

After processing a video:

```powershell
python make_demo.py --input output/results.avi --gif assets/demo.gif --preview assets/preview.jpg
```

Review the generated files for privacy before committing them. Then add them to this README:

```markdown
![Detection preview](assets/preview.jpg)
![Tracking demonstration](assets/demo.gif)
```

## Configuration

Important values in `config.ini`:

- `weights`: path to the trained YOLOv7 weights
- `imgsz`: inference image size
- `conf_thres`: confidence threshold
- `iou_thres`: non-maximum suppression IoU threshold
- `device`: `cpu` or a CUDA device such as `0`

The default restricted area is defined as relative coordinates in `draw.py`, so it scales to the input resolution. Adjust `ImageDrawer.DEFAULT_POINTS` for a different camera view.

## Model classes and quantitative performance

See [MODEL_CARD.md](MODEL_CARD.md). The repository currently does not contain the checkpoint's verified class-name file or test-set evaluation output, so class names and mAP values are deliberately marked as unreported rather than guessed.

Before presenting this project, update the model card with:

- Exact class ID order used during training
- Precision and recall
- mAP@0.5 and mAP@0.5:0.95
- Test-set size and class distribution
- Hardware and measured FPS
- Representative successful and failed detections

## Limitations

- Model weights and demonstration videos must be supplied separately.
- Run `calibrate_roi.py` once for each camera angle and keep its ROI JSON with that camera configuration.
- Detection quality depends on the training data, lighting, occlusion and camera placement.
- Warning classes are configurable with repeatable `--class-id` options; the default is class ID 0. Verify the correct person-class ID for your trained model.

## License and attribution

This repository is distributed under the GNU General Public License v3.0 or later. See [LICENSE](LICENSE).

- YOLOv7 inference portions are based on the [official YOLOv7 project](https://github.com/WongKinYiu/yolov7), which is licensed under GPL-3.0.

## Privacy and safety

Do not use this project as the only safety control on a construction site. Follow applicable privacy, workplace-monitoring and occupational-safety requirements.
