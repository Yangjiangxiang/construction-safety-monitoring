import argparse

from Stream import VideoTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv7 construction-site safety monitoring"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Path to an input video")
    source.add_argument("--camera", type=int, help="Camera index, for example 0")
    parser.add_argument(
        "--output",
        default="output",
        help="Directory used to save results.avi",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process without opening a preview window",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Process every Nth frame",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    camera = args.camera if args.camera is not None else -1
    with VideoTracker(
        cam=camera,
        video_path=args.video or "",
        save_path=args.output,
        use_frame=(0, 1),
        display=not args.no_display,
        frame_interval=args.frame_interval,
    ) as tracker:
        tracker.run()


if __name__ == "__main__":
    main()
