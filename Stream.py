# Construction Safety Monitoring
# Copyright (C) 2026 Yangjiangxiang (modifications and project integration)
# Portions are based on YOLOv7 and an earlier third-party example.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This modified file is distributed without warranty. See LICENSE and
# THIRD_PARTY_NOTICES.md for license terms and attribution.

import configparser
import os
import sys
import time

import cv2
from loguru import logger

from draw import ImageDrawer
from project_yolov7det import yolov7Pt_infer
from tracking import CentroidTracker, EventRecorder

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VideoTracker:
    def __init__(
        self,
        cam=-1,
        video_path="",
        save_path="",
        use_frame=(0, 1),
        display=True,
        frame_interval=1,
        weights_path=None,
        restricted_area_points=None,
        allowed_class_ids=None,
        tracker_max_distance=100,
        tracker_max_missed=20,
    ):
        self.display = display
        self.use_frame = use_frame
        self.video_path = video_path
        self.cam = cam
        self.save_path = save_path
        self.frame_interval = max(1, int(frame_interval))
        self.weights_path = weights_path
        self.allowed_class_ids = (
            None if allowed_class_ids is None else set(allowed_class_ids)
        )
        self.tracker = CentroidTracker(
            max_distance=tracker_max_distance,
            max_missed=tracker_max_missed,
        )
        self.event_recorder = None

        if self.cam != -1:
            logger.info("Using webcam: {}", self.cam)
            self.vdo = cv2.VideoCapture(self.cam)
        else:
            logger.info("Using video: {}", self.video_path)
            self.vdo = cv2.VideoCapture()

        self.det = yolov7Pt_infer(*self.get_pt_model_config(self.weights_path))
        self.draw_tool = ImageDrawer(restricted_area_points)

    def __enter__(self):
        if self.cam != -1:
            if not self.vdo.isOpened():
                raise RuntimeError(f"Unable to open camera {self.cam}")
            ret, frame = self.vdo.read()
            if not ret:
                raise RuntimeError("Unable to read a frame from the camera")
            self.im_height, self.im_width = frame.shape[:2]
            self.count_frame = -1
        else:
            if not os.path.isfile(self.video_path):
                raise FileNotFoundError(
                    f"Input video not found: {self.video_path}. "
                    "Provide a valid file with --video."
                )
            self.vdo.open(self.video_path)
            if not self.vdo.isOpened():
                raise RuntimeError(f"Unable to open video: {self.video_path}")
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.count_frame = int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT))

        self.source_fps = self.vdo.get(cv2.CAP_PROP_FPS)
        if not self.source_fps or self.source_fps <= 0:
            self.source_fps = 24

        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            self.event_recorder = EventRecorder(self.save_path)
            self.save_video_path = os.path.join(self.save_path, "results.avi")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = cv2.VideoWriter(
                self.save_video_path,
                fourcc,
                self.source_fps,
                (self.im_width, self.im_height),
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Unable to create output video: {self.save_video_path}")
            logger.info("Saving results to {}", self.save_video_path)

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if hasattr(self, "vdo") and self.vdo.isOpened():
            self.vdo.release()
        if hasattr(self, "writer"):
            self.writer.release()
        if self.display:
            cv2.destroyAllWindows()
        return False

    def run(self):
        idx_frame = 0
        total_elapsed = 0.0

        if self.count_frame > 0:
            start_frame = int(self.count_frame * self.use_frame[0])
            end_frame = int(self.count_frame * self.use_frame[1])
        else:
            start_frame = 0
            end_frame = None

        while self.vdo.grab():
            idx_frame += 1

            if idx_frame % self.frame_interval:
                continue
            if idx_frame < start_frame:
                continue
            if end_frame is not None and idx_frame > end_frame:
                break

            started_at = time.perf_counter()
            retrieved, original = self.vdo.retrieve()
            if not retrieved:
                continue

            annotated, _, predictions = self.det.infer(original.copy())
            annotated = self.draw_tool.draw_polygon(annotated)

            filtered = [
                detection
                for detection in predictions
                if (
                    self.allowed_class_ids is None
                    or int(detection[5]) in self.allowed_class_ids
                )
            ]
            tracks, removed_tracks = self.tracker.update(filtered)
            video_time = idx_frame / self.source_fps

            for track in removed_tracks:
                if track.inside and self.event_recorder:
                    self.event_recorder.record(
                        track,
                        "lost_in_zone",
                        idx_frame,
                        video_time,
                    )

            for track in tracks:
                xmin, ymin, xmax, ymax = track.bbox
                foot_x = (xmin + xmax) // 2
                foot_y = ymax
                is_inside = self.draw_tool.is_inside_polygon(
                    annotated,
                    foot_x,
                    foot_y,
                )

                event = None
                if is_inside and not track.inside:
                    event = "enter"
                elif not is_inside and track.inside:
                    event = "exit"
                track.inside = is_inside

                annotated = self.draw_tool.draw_tracked_box(
                    annotated,
                    track.bbox,
                    track.track_id,
                    track.class_id,
                    track.confidence,
                    is_inside,
                )

                if event and self.event_recorder:
                    snapshot = ""
                    if event == "enter":
                        snapshot_name = (
                            f"frame_{idx_frame:08d}_track_{track.track_id}.jpg"
                        )
                        snapshot = os.path.join("events", snapshot_name)
                        cv2.imwrite(
                            os.path.join(self.save_path, snapshot),
                            annotated,
                        )
                    self.event_recorder.record(
                        track,
                        event,
                        idx_frame,
                        video_time,
                        snapshot,
                    )

            if self.display:
                cv2.imshow("Construction Safety Monitoring", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if hasattr(self, "writer"):
                self.writer.write(annotated)

            elapsed = max(time.perf_counter() - started_at, 1e-9)
            total_elapsed += elapsed
            fps = 1 / elapsed

            if self.display:
                total_label = end_frame if end_frame is not None else "live"
                logger.info(
                    "Frame {}/{}: {:.2f} ms, FPS {:.2f}",
                    idx_frame,
                    total_label,
                    elapsed * 1000,
                    fps,
                )
            elif end_frame:
                self.print_progress_bar(idx_frame, end_frame)

        logger.info("Total inference time: {:.3f} s", total_elapsed)

    @staticmethod
    def print_progress_bar(iteration, total, length=50):
        if total <= 0:
            return
        ratio = min(max(iteration / total, 0), 1)
        filled_length = int(length * ratio)
        bar = "█" * filled_length + "-" * (length - filled_length)
        sys.stdout.write(f"\rProgress: |{bar}| {ratio * 100:.1f}% Complete")
        sys.stdout.flush()

    @staticmethod
    def get_pt_model_config(weights_override=None):
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), "config.ini")
        if not config.read(config_path, encoding="utf-8"):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        section = config["config"]
        configured_weights = weights_override or section.get("weights")
        if os.path.isabs(configured_weights):
            weights = os.path.normpath(configured_weights)
        else:
            weights = os.path.normpath(
                os.path.join(os.path.dirname(__file__), configured_weights)
            )
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"Model weights not found: {weights}. See README.md for setup."
            )

        classes_text = section.get("classes", "None")
        classes = (
            None
            if classes_text.lower() == "none"
            else [int(value) for value in classes_text.split()]
        )

        return (
            weights,
            section.getint("imgsz"),
            section.getfloat("conf_thres"),
            section.getfloat("iou_thres"),
            section.get("device"),
            section.getboolean("save_conf"),
            section.getboolean("nosave"),
            classes,
            section.getboolean("agnostic_nms"),
            section.getboolean("augment"),
            section.getboolean("update"),
            section.getboolean("no_trace"),
        )
