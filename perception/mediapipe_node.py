"""ROS 2 node that publishes MediaPipe pose world landmarks and camera frames."""

from __future__ import annotations

import math
import os
import time
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from one_euro_filter import OneEuroFilter


LANDMARK_COUNT = 33
VALUES_PER_LANDMARK = 4
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

try:
    from cv_bridge import CvBridge

    CV_BRIDGE_AVAILABLE = True
except Exception:
    CvBridge = None
    CV_BRIDGE_AVAILABLE = False


def _resolve_mediapipe_solutions():
    solutions = getattr(mp, "solutions", None)
    if solutions is not None:
        return solutions

    from mediapipe.python import solutions as mp_solutions

    return mp_solutions


class MediaPipeNode(Node):
    """Capture webcam frames, estimate pose, and publish ROS topics."""

    def __init__(self) -> None:
        super().__init__("mediapipe_node")

        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        self.declare_parameter("show_debug_window", has_display)
        self._show_debug_window = (
            self.get_parameter("show_debug_window").get_parameter_value().bool_value
        )

        if self._show_debug_window and not has_display:
            self.get_logger().warning(
                "show_debug_window requested but no desktop display found; disabling preview window"
            )
            self._show_debug_window = False

        self.bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None
        self.pose_publisher = self.create_publisher(
            Float32MultiArray,
            "/mediapipe/pose_world_landmarks",
            10,
        )
        self.image_publisher = self.create_publisher(Image, "/camera/image_raw", 10)

        if not CV_BRIDGE_AVAILABLE:
            self.get_logger().warning(
                "cv_bridge unavailable; /camera/image_raw publishing disabled (landmarks continue)"
            )

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not self.capture.isOpened():
            raise RuntimeError("Unable to open webcam on cv2.VideoCapture(0)")

        mp_solutions = _resolve_mediapipe_solutions()
        self.mp_pose = mp_solutions.pose
        self.mp_drawing = mp_solutions.drawing_utils
        self.mp_drawing_styles = mp_solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.coordinate_filters: List[List[OneEuroFilter]] = [
            [OneEuroFilter(min_cutoff=1.0, beta=0.01) for _ in range(3)]
            for _ in range(LANDMARK_COUNT)
        ]

        self.timer = self.create_timer(1.0 / 30.0, self.process_frame)
        self.get_logger().info("mediapipe_node started")

    def process_frame(self) -> None:
        success, frame = self.capture.read()
        if not success:
            self.get_logger().warning("Failed to read frame from webcam")
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        landmark_msg = Float32MultiArray()
        landmark_msg.data = self._build_landmark_payload(results)
        self.pose_publisher.publish(landmark_msg)

        annotated_frame = self._annotate_frame(frame, results)

        if self._show_debug_window:
            cv2.imshow("MediaPipe Pose", annotated_frame)
            cv2.waitKey(1)

        if self.bridge is not None:
            image_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = "camera"
            self.image_publisher.publish(image_msg)

    def _build_landmark_payload(self, results) -> List[float]:
        data = [math.nan] * (LANDMARK_COUNT * VALUES_PER_LANDMARK)

        if not results.pose_world_landmarks:
            for index in range(LANDMARK_COUNT):
                data[index * VALUES_PER_LANDMARK + 3] = 0.0
            return data

        timestamp = time.monotonic()
        world_landmarks = results.pose_world_landmarks.landmark
        image_landmarks = (
            results.pose_landmarks.landmark if results.pose_landmarks else None
        )

        for index, landmark in enumerate(world_landmarks[:LANDMARK_COUNT]):
            filtered_x = self.coordinate_filters[index][0](timestamp, landmark.x)
            filtered_y = self.coordinate_filters[index][1](timestamp, landmark.y)
            filtered_z = self.coordinate_filters[index][2](timestamp, landmark.z)

            visibility = landmark.visibility
            if image_landmarks is not None and index < len(image_landmarks):
                visibility = image_landmarks[index].visibility

            base = index * VALUES_PER_LANDMARK
            data[base] = float(filtered_x)
            data[base + 1] = float(filtered_y)
            data[base + 2] = float(filtered_z)
            data[base + 3] = float(visibility)

        return data

    def _annotate_frame(self, frame: np.ndarray, results) -> np.ndarray:
        annotated = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        return annotated

    def destroy_node(self) -> bool:
        if hasattr(self, "capture") and self.capture is not None:
            self.capture.release()
        if hasattr(self, "pose") and self.pose is not None:
            self.pose.close()
        cv2.destroyAllWindows()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MediaPipeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
