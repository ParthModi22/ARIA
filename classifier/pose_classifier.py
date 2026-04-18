"""ROS 2 node that classifies poses from MediaPipe pose world landmarks."""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, String
except ImportError:
    rclpy = None
    Node = object
    Float32MultiArray = None
    String = None


EXPECTED_FLOAT_COUNT = 33 * 4
LANDMARK_COUNT = 33
POSE_LABELS = ["T_POSE", "HANDS_UP", "LEFT_ARM_UP", "RIGHT_ARM_UP", "SQUAT"]
PAIRWISE_JOINTS = (
    (15, 11),
    (16, 12),
    (15, 12),
    (16, 11),
    (11, 23),
    (12, 24),
    (25, 23),
    (26, 24),
    (27, 23),
    (28, 24),
    (15, 23),
    (16, 24),
    (15, 16),
    (25, 26),
    (15, 0),
    (16, 0),
    (25, 27),
    (26, 28),
    (23, 24),
    (11, 12),
    (13, 23),
    (14, 24),
    (13, 11),
    (14, 12),
)


def extract_features(landmarks: np.ndarray) -> np.ndarray:
    """Extract normalized distance features from pose landmarks."""
    if landmarks.shape != (LANDMARK_COUNT, 3):
        raise ValueError(f"Expected landmarks shape {(LANDMARK_COUNT, 3)}, got {landmarks.shape}")

    torso_left = np.linalg.norm(landmarks[11] - landmarks[23])
    torso_right = np.linalg.norm(landmarks[12] - landmarks[24])
    torso_size = (torso_left + torso_right) / 2.0
    if torso_size <= 1e-9 or np.isnan(torso_size):
        torso_size = 1.0

    features = [
        np.linalg.norm(landmarks[first] - landmarks[second]) / torso_size
        for first, second in PAIRWISE_JOINTS
    ]
    return np.asarray(features, dtype=np.float32)


class EmaDebounce:
    """Frame-based stable label selector with a confidence threshold."""

    def __init__(self, window: int = 8, confidence_threshold: float = 0.80) -> None:
        self.window = window
        self.confidence_threshold = confidence_threshold
        self.history: deque[tuple[str, float]] = deque(maxlen=window)

    def update(self, label: str, confidence: float) -> str | None:
        if confidence < self.confidence_threshold:
            self.history.clear()
            return None

        self.history.append((label, confidence))
        if len(self.history) < self.window:
            return None

        labels = [entry[0] for entry in self.history]
        if all(entry == labels[0] for entry in labels):
            return labels[0]
        return None


class PoseClassifierNode(Node):
    """Subscribe to pose landmarks and publish raw and stable pose labels."""

    def __init__(self) -> None:
        super().__init__("pose_classifier_node")

        self.raw_publisher = self.create_publisher(String, "/pose_classifier/raw_pose", 10)
        self.stable_publisher = self.create_publisher(
            String, "/pose_classifier/current_pose", 10
        )
        self.subscription = self.create_subscription(
            Float32MultiArray,
            "/mediapipe/pose_world_landmarks",
            self.landmarks_callback,
            10,
        )

        self.model = self._load_model()
        self.debounce = EmaDebounce(window=8, confidence_threshold=0.80)
        self.last_stable_pose: str | None = None

        self.get_logger().info("pose_classifier_node started")

    def _load_model(self):
        model_path = Path(__file__).resolve().parent / "models" / "knn_pose_model.pkl"
        if not model_path.exists():
            self.get_logger().warning(
                f"Model file missing at {model_path}; publishing UNKNOWN every frame"
            )
            return None

        try:
            with model_path.open("rb") as stream:
                model = pickle.load(stream)
        except Exception as exc:
            self.get_logger().warning(
                f"Failed to load model from {model_path}: {exc}; publishing UNKNOWN every frame"
            )
            return None

        self.get_logger().info(f"Loaded classifier model from {model_path}")
        return model

    def landmarks_callback(self, msg: Float32MultiArray) -> None:
        landmarks = self._unpack_landmarks(msg.data)

        if landmarks is None:
            return

        if self.model is None:
            self._publish_raw("UNKNOWN", 0.0)
            self._publish_stable("UNKNOWN")
            return

        features = extract_features(landmarks).reshape(1, -1)
        label, confidence = self._predict(features)

        self._publish_raw(label, confidence)

        stable_label = self.debounce.update(label, confidence)
        if stable_label is not None:
            self._publish_stable(stable_label)

    def _unpack_landmarks(self, flat_data) -> np.ndarray | None:
        if len(flat_data) != EXPECTED_FLOAT_COUNT:
            self.get_logger().warning(
                f"Expected {EXPECTED_FLOAT_COUNT} floats, received {len(flat_data)}"
            )
            return None

        xyz = np.asarray(flat_data, dtype=np.float32).reshape(LANDMARK_COUNT, 4)[:, :3]
        if np.isnan(xyz).any():
            return None
        return xyz

    def _predict(self, features: np.ndarray) -> tuple[str, float]:
        if hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(features)[0], dtype=np.float32)
            best_index = int(np.argmax(probabilities))
            label = str(self.model.classes_[best_index])
            confidence = float(probabilities[best_index])
            return label, confidence

        label = str(self.model.predict(features)[0])
        return label, 1.0

    def _publish_raw(self, label: str, confidence: float) -> None:
        raw_msg = String()
        raw_msg.data = f"{label}:{confidence:.2f}"
        self.raw_publisher.publish(raw_msg)

    def _publish_stable(self, label: str) -> None:
        stable_msg = String()
        stable_msg.data = label
        self.stable_publisher.publish(stable_msg)

        if label != self.last_stable_pose:
            self.get_logger().info(
                f"Pose transition: {self.last_stable_pose or 'NONE'} -> {label}"
            )
            self.last_stable_pose = label


def main(args: list[str] | None = None) -> None:
    if rclpy is None:
        raise RuntimeError("ROS2 Python dependencies are not available in this environment")

    rclpy.init(args=args)
    node = PoseClassifierNode()

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
