"""Create a compact GIF and preview image from an annotated result video."""

import argparse

import cv2
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GitHub demo media")
    parser.add_argument("--input", default="output/results.avi")
    parser.add_argument("--gif", default="assets/demo.gif")
    parser.add_argument("--preview", default="assets/preview.jpg")
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--gif-fps", type=float, default=6)
    parser.add_argument("--width", type=int, default=800)
    return parser.parse_args()


def main():
    args = parse_args()
    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {args.input}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 24
    interval = max(1, round(source_fps / args.gif_fps))
    limit = max(1, round(args.seconds * args.gif_fps))
    frames = []
    frame_number = 0

    while len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_number % interval == 0:
            height, width = frame.shape[:2]
            output_height = round(height * args.width / width)
            frame = cv2.resize(frame, (args.width, output_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        frame_number += 1
    capture.release()

    if not frames:
        raise RuntimeError("No frames were read from the result video")

    import os
    for path in (args.gif, args.preview):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    frames[len(frames) // 2].save(args.preview, quality=90)
    duration_ms = round(1000 / args.gif_fps)
    frames[0].save(
        args.gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"Saved {args.gif} and {args.preview}")


if __name__ == "__main__":
    main()
