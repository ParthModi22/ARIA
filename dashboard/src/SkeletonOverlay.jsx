import { useEffect, useRef } from "react";

const CONNECTIONS = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28],
];

const VALUES_PER_LANDMARK = 4;
const LANDMARK_COUNT = 33;
const MIN_VISIBILITY = 0.35;

export default function SkeletonOverlay({
  landmarks,
  width = 1280,
  height = 720,
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    context.clearRect(0, 0, width, height);

    if (
      !Array.isArray(landmarks) ||
      landmarks.length < LANDMARK_COUNT * VALUES_PER_LANDMARK
    ) {
      return;
    }

    const points = [];
    for (let index = 0; index < LANDMARK_COUNT; index += 1) {
      const base = index * VALUES_PER_LANDMARK;
      const x = Number(landmarks[base]);
      const y = Number(landmarks[base + 1]);
      const visibility = Number(landmarks[base + 3]);

      if (
        !Number.isFinite(x) ||
        !Number.isFinite(y) ||
        !Number.isFinite(visibility) ||
        visibility < MIN_VISIBILITY
      ) {
        points.push(null);
        continue;
      }

      points.push({
        x: Math.max(0, Math.min(1, x)) * width,
        y: Math.max(0, Math.min(1, y)) * height,
      });
    }

    context.strokeStyle = "#00ff00";
    context.lineWidth = 2;

    CONNECTIONS.forEach(([startIndex, endIndex]) => {
      const start = points[startIndex];
      const end = points[endIndex];
      if (!start || !end) {
        return;
      }

      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();
    });

    context.fillStyle = "#00ff00";
    points.forEach((point) => {
      if (!point) {
        return;
      }

      context.beginPath();
      context.arc(point.x, point.y, 4, 0, Math.PI * 2);
      context.fill();
    });
  }, [height, landmarks, width]);

  return (
    <canvas
      ref={canvasRef}
      aria-label="Skeleton overlay"
      width={width}
      height={height}
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
      }}
    />
  );
}
