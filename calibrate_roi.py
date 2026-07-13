import argparse
import json

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Click at least three points to calibrate a restricted area"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Video used for camera calibration")
    source.add_argument("--camera", type=int, help="Camera index")
    parser.add_argument("--output", default="roi.json", help="Output ROI JSON")
    return parser.parse_args()


def read_reference_frame(video=None, camera=None):
    capture = cv2.VideoCapture(camera if camera is not None else video)
    if not capture.isOpened():
        raise RuntimeError("Unable to open the calibration source")
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Unable to read a calibration frame")
    return frame


def main():
    args = parse_args()
    frame = read_reference_frame(args.video, args.camera)
    preview = frame.copy()
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    window = "ROI calibration"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        preview = frame.copy()
        if points:
            for point in points:
                cv2.circle(preview, point, 5, (0, 0, 255), -1)
            if len(points) >= 2:
                cv2.polylines(
                    preview,
                    [__import__("numpy").array(points, dtype="int32")],
                    len(points) >= 3,
                    (0, 0, 255),
                    2,
                )

        cv2.putText(
            preview,
            "Left click: add | Right click: undo | Enter: save | Esc: cancel",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10) and len(points) >= 3:
            break
        if key == 27:
            cv2.destroyAllWindows()
            return

    height, width = frame.shape[:2]
    data = {
        "normalized_points": [
            [round(x / width, 6), round(y / height, 6)]
            for x, y in points
        ],
        "reference_size": {"width": width, "height": height},
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    cv2.destroyAllWindows()
    print(f"Saved restricted area to {args.output}")


if __name__ == "__main__":
    main()
