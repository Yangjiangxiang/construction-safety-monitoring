"""Lightweight centroid-based tracking and CSV safety-event recording."""

import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Track:
    track_id: int
    bbox: tuple
    class_id: int
    confidence: float
    missed: int = 0
    inside: bool = False

    @property
    def center(self):
        xmin, ymin, xmax, ymax = self.bbox
        return ((xmin + xmax) / 2, (ymin + ymax) / 2)


class CentroidTracker:
    """Assign stable IDs using greedy same-class centroid matching."""

    def __init__(self, max_distance=100, max_missed=20):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def _distance(track, detection):
        xmin, ymin, xmax, ymax, _, _ = detection
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        tx, ty = track.center
        return math.hypot(cx - tx, cy - ty)

    def update(self, detections):
        unmatched_track_ids = set(self.tracks)
        unmatched_detection_ids = set(range(len(detections)))
        candidates = []

        for track_id, track in self.tracks.items():
            for detection_id, detection in enumerate(detections):
                class_id = int(detection[5])
                if class_id != track.class_id:
                    continue
                distance = self._distance(track, detection)
                if distance <= self.max_distance:
                    candidates.append((distance, track_id, detection_id))

        for _, track_id, detection_id in sorted(candidates):
            if (
                track_id not in unmatched_track_ids
                or detection_id not in unmatched_detection_ids
            ):
                continue
            xmin, ymin, xmax, ymax, score, class_id = detections[detection_id]
            track = self.tracks[track_id]
            track.bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
            track.confidence = float(score)
            track.class_id = int(class_id)
            track.missed = 0
            unmatched_track_ids.remove(track_id)
            unmatched_detection_ids.remove(detection_id)

        for track_id in unmatched_track_ids:
            self.tracks[track_id].missed += 1

        removed = []
        for track_id in list(self.tracks):
            if self.tracks[track_id].missed > self.max_missed:
                removed.append(self.tracks.pop(track_id))

        for detection_id in unmatched_detection_ids:
            xmin, ymin, xmax, ymax, score, class_id = detections[detection_id]
            track = Track(
                track_id=self.next_id,
                bbox=(int(xmin), int(ymin), int(xmax), int(ymax)),
                confidence=float(score),
                class_id=int(class_id),
            )
            self.tracks[self.next_id] = track
            self.next_id += 1

        visible = [track for track in self.tracks.values() if track.missed == 0]
        return visible, removed


class EventRecorder:
    COLUMNS = [
        "timestamp_utc",
        "video_time_seconds",
        "frame",
        "track_id",
        "event",
        "class_id",
        "confidence",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "snapshot",
    ]

    def __init__(self, output_directory):
        self.output_directory = output_directory
        self.snapshot_directory = os.path.join(output_directory, "events")
        self.csv_path = os.path.join(output_directory, "events.csv")
        os.makedirs(self.snapshot_directory, exist_ok=True)

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as file:
                csv.DictWriter(file, fieldnames=self.COLUMNS).writeheader()

    def record(self, track, event, frame_number, video_time, snapshot=""):
        xmin, ymin, xmax, ymax = track.bbox
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "video_time_seconds": round(video_time, 3),
            "frame": frame_number,
            "track_id": track.track_id,
            "event": event,
            "class_id": track.class_id,
            "confidence": round(track.confidence, 5),
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "snapshot": snapshot,
        }
        with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as file:
            csv.DictWriter(file, fieldnames=self.COLUMNS).writerow(row)
