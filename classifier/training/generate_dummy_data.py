"""Generate synthetic pose samples, save CSVs, and train the KNN classifier."""

from __future__ import annotations

import csv
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classifier.pose_classifier import POSE_LABELS, extract_features
from classifier.training.train_knn import build_pipeline, load_dataset


SAMPLES_PER_POSE = 200
LANDMARK_COUNT = 33
NOISE_STD = 0.05


def _base_landmarks() -> np.ndarray:
    landmarks = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)

    landmarks[0] = [0.0, -0.25, 0.0]
    landmarks[11] = [-0.2, 0.0, 0.0]
    landmarks[12] = [0.2, 0.0, 0.0]
    landmarks[13] = [-0.35, 0.0, 0.0]
    landmarks[14] = [0.35, 0.0, 0.0]
    landmarks[15] = [-0.5, 0.0, 0.0]
    landmarks[16] = [0.5, 0.0, 0.0]
    landmarks[23] = [-0.15, 0.4, 0.0]
    landmarks[24] = [0.15, 0.4, 0.0]
    landmarks[25] = [-0.15, 0.8, 0.0]
    landmarks[26] = [0.15, 0.8, 0.0]
    landmarks[27] = [-0.15, 1.2, 0.0]
    landmarks[28] = [0.15, 1.2, 0.0]

    return landmarks


def pose_template(pose_name: str) -> np.ndarray:
    landmarks = _base_landmarks()

    if pose_name == "T_POSE":
        landmarks[15] = [-0.5, 0.0, 0.0]
        landmarks[16] = [0.5, 0.0, 0.0]
    elif pose_name == "HANDS_UP":
        landmarks[13] = [-0.2, -0.3, 0.0]
        landmarks[14] = [0.2, -0.3, 0.0]
        landmarks[15] = [-0.2, -0.6, 0.0]
        landmarks[16] = [0.2, -0.6, 0.0]
    elif pose_name == "LEFT_ARM_UP":
        landmarks[13] = [-0.25, -0.2, 0.0]
        landmarks[15] = [-0.2, -0.5, 0.0]
        landmarks[16] = [0.45, 0.2, 0.0]
    elif pose_name == "RIGHT_ARM_UP":
        landmarks[14] = [0.25, -0.2, 0.0]
        landmarks[16] = [0.2, -0.5, 0.0]
        landmarks[15] = [-0.45, 0.2, 0.0]
    elif pose_name == "SQUAT":
        landmarks[25] = [-0.15, 0.3, 0.0]
        landmarks[26] = [0.15, 0.3, 0.0]
        landmarks[27] = [-0.18, 0.6, 0.0]
        landmarks[28] = [0.18, 0.6, 0.0]
        landmarks[15] = [-0.3, 0.1, 0.0]
        landmarks[16] = [0.3, 0.1, 0.0]
    else:
        raise ValueError(f"Unsupported pose label: {pose_name}")

    return landmarks


def generate_samples(pose_name: str, sample_count: int) -> list[list[float]]:
    template = pose_template(pose_name)
    rows: list[list[float]] = []

    for _ in range(sample_count):
        noisy_landmarks = template + np.random.normal(
            loc=0.0,
            scale=NOISE_STD,
            size=template.shape,
        ).astype(np.float32)
        features = extract_features(noisy_landmarks)
        rows.append([pose_name] + features.tolist())

    return rows


def save_pose_csv(data_dir: Path, pose_name: str, rows: list[list[float]]) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = data_dir / f"{pose_name}_{timestamp}.csv"

    header = ["label"] + [f"f{index}" for index in range(len(rows[0]) - 1)]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)

    return output_path


def train_and_save_model(data_dir: Path, model_path: Path) -> float:
    x, y = load_dataset(data_dir)
    pipeline = build_pipeline()

    scores = cross_val_score(pipeline, x, y, cv=5)
    pipeline.fit(x, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as stream:
        pickle.dump(pipeline, stream)

    return float(scores.mean())


def main() -> None:
    training_dir = Path(__file__).resolve().parent
    data_dir = training_dir / "data"
    model_path = training_dir.parent / "models" / "knn_pose_model.pkl"

    for pose_name in POSE_LABELS:
        rows = generate_samples(pose_name, SAMPLES_PER_POSE)
        output_path = save_pose_csv(data_dir, pose_name, rows)
        print(f"Saved {len(rows)} samples to {output_path}")

    accuracy = train_and_save_model(data_dir, model_path)
    print(f"Final cross-validation accuracy: {accuracy:.4f}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
