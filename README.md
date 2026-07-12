# Construction Safety Monitoring

A YOLOv7-based construction-site monitoring project for safety detection and restricted-area intrusion warnings.

## Features

- Video-file and webcam input
- YOLOv7 object detection
- Resolution-independent restricted-area overlay
- Intrusion warning based on the bottom-center point of each detection
- Annotated video export
- CPU or CUDA device configuration

## Project structure

- `main.py` — command-line entry point
- `Stream.py` — video capture, inference and output pipeline
- `draw.py` — restricted-area drawing and intrusion checks
- `config.ini` — model and inference settings
- `project_yolov7det/` — YOLOv7 inference implementation

## Model weights

The trained file `best.pt` is not stored in this repository because it is larger than GitHub's 100 MB file limit.

Place the model at:

```text
project_yolov7det/weights/pt/best.pt
```

The expected location can be changed in `config.ini`.

> Only load model files from a source you trust. PyTorch weight files can contain executable serialized content.

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

Process a video:

```powershell
python main.py --video "path\to\input.mp4" --output output
```

Use webcam 0:

```powershell
python main.py --camera 0 --output output
```

Process without opening a preview window:

```powershell
python main.py --video "path\to\input.mp4" --no-display
```

Press `q` to stop the preview. The annotated result is saved as `output/results.avi`.

## Configuration

Important values in `config.ini`:

- `weights`: path to the trained YOLOv7 weights
- `imgsz`: inference image size
- `conf_thres`: confidence threshold
- `iou_thres`: non-maximum suppression IoU threshold
- `device`: `cpu` or a CUDA device such as `0`

The default restricted area is defined as relative coordinates in `draw.py`, so it scales to the input resolution. Adjust `ImageDrawer.DEFAULT_POINTS` for a different camera view.

## Limitations

- Model weights and demonstration videos must be supplied separately.
- Restricted-area coordinates must be calibrated for each camera angle.
- Detection quality depends on the training data, lighting, occlusion and camera placement.
- The current warning logic applies to every returned detection. Configure the model classes when only person detections should be considered.

## Attribution

This project includes a YOLOv7 inference implementation and code adapted from earlier third-party examples. Before redistributing or using it commercially, verify and preserve the licenses and attribution requirements of the original YOLOv7 implementation and any other incorporated source code.

## Privacy and safety

Do not use this project as the only safety control on a construction site. Follow applicable privacy, workplace-monitoring and occupational-safety requirements.
