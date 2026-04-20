"""Train and export a KNN pose classifier from collected CSV samples."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    features: list[list[float]] = []
    labels: list[str] = []

    for csv_path in csv_paths:
        with csv_path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if header is None:
                continue

            for row in reader:
                if not row:
                    continue
                labels.append(row[0])
                features.append([float(value) for value in row[1:]])

    if not features:
        raise ValueError(f"No sample rows found in {data_dir}")

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=object)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ]
    )


def main() -> None:
    training_dir = Path(__file__).resolve().parent
    data_dir = training_dir / "data"
    model_path = training_dir.parent / "models" / "knn_pose_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    x, y = load_dataset(data_dir)
    pipeline = build_pipeline()

    scores = cross_val_score(pipeline, x, y, cv=5)
    print(f"Cross-validation accuracy (5-fold): {scores.mean():.4f} +/- {scores.std():.4f}")

    pipeline.fit(x, y)

    with model_path.open("wb") as stream:
        pickle.dump(pipeline, stream)

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
