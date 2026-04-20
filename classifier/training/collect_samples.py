"""Collect labeled pose samples from a webcam into CSV files."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classifier.pose_classifier import extract_features


POSE_GUIDES = {
    "T_POSE": "Stand tall with both arms straight out to the sides at shoulder height.",
    "HANDS_UP": "Stand tall and lift both hands straight above your head.",
    "LEFT_ARM_UP": "Keep your right arm down and raise your left arm straight overhead.",
    "RIGHT_ARM_UP": "Keep your left arm down and raise your right arm straight overhead.",
    "SQUAT": "Bend both knees into a shallow squat while keeping your chest upright.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect pose classification samples")
    parser.add_argument(
        "--pose",
        required=True,
        choices=sorted(POSE_GUIDES.keys()),
        help="Pose label to record",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples to collect",
    )
    return parser.parse_args()


def wrap_text(text: str, width: int = 42) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def draw_overlay(frame: np.ndarray, pose_name: str, captured: int, total: int) -> None:
    cv2.rectangle(frame, (10, 10), (700, 145), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"{pose_name}: {captured}/{total}",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    y = 74
    for line in wrap_text(POSE_GUIDES[pose_name]):
        cv2.putText(
            frame,
            line,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24

    cv2.putText(
        frame,
        "SPACE: capture sample    Q: quit",
        (24, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 220, 0),
        1,
        cv2.LINE_AA,
    )


def save_samples(pose_name: str, rows: list[list[float]]) -> Path:
    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{pose_name}_{timestamp}.csv"

    header = ["label"] + [f"f{index}" for index in range(len(rows[0]) - 1)]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)

    return output_path


def main() -> None:
    args = parse_args()

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Unable to open webcam on cv2.VideoCapture(0)")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    pose = mp_pose.Pose(
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    collected_rows: list[list[float]] = []

    try:
        while len(collected_rows) < args.samples:
            success, frame = capture.read()
            if not success:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            display_frame = frame.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

            draw_overlay(display_frame, args.pose, len(collected_rows), args.samples)
            cv2.imshow("Pose Sample Collector", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord(" "):
                if not results.pose_world_landmarks:
                    continue

                world_landmarks = results.pose_world_landmarks.landmark
                if len(world_landmarks) != 33:
                    continue

                landmarks = np.asarray(
                    [[landmark.x, landmark.y, landmark.z] for landmark in world_landmarks],
                    dtype=np.float32,
                )
                features = extract_features(landmarks)
                row = [args.pose] + features.astype(np.float32).tolist()
                collected_rows.append(row)

        if collected_rows:
            output_path = save_samples(args.pose, collected_rows)
            print(f"Saved {len(collected_rows)} samples to {output_path}")
        else:
            print("No samples collected.")
    finally:
        pose.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
