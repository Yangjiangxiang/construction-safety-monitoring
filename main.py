import argparse
import json
import os

from Stream import VideoTracker


def load_roi(path):
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"ROI file not found: {path}. Run calibrate_roi.py first."
        )
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    points = data.get("normalized_points")
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("ROI file must contain at least three normalized_points")
    return [tuple(map(float, point)) for point in points]


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv7 construction-site safety monitoring"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Path to an input video")
    source.add_argument("--camera", type=int, help="Camera index, for example 0")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument(
        "--weights",
        help="Override the model weights path from config.ini",
    )
    parser.add_argument(
        "--roi",
        default="roi.json",
        help="Camera-specific ROI JSON created by calibrate_roi.py",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        action="append",
        dest="class_ids",
        help="Class ID allowed to trigger warnings; repeat for multiple IDs. Default: 0",
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--frame-interval", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    camera = args.camera if args.camera is not None else -1
    class_ids = args.class_ids if args.class_ids is not None else [0]

    with VideoTracker(
        cam=camera,
        video_path=args.video or "",
        save_path=args.output,
        display=not args.no_display,
        frame_interval=args.frame_interval,
        weights_path=args.weights,
        restricted_area_points=load_roi(args.roi),
        allowed_class_ids=class_ids,
    ) as tracker:
        tracker.run()


if __name__ == "__main__":
    main()
