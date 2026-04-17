import { useEffect, useState } from "react";
import ROSLIB from "roslib";

import SkeletonOverlay from "./SkeletonOverlay";

const ROS_WS_URL =
  (typeof process !== "undefined" && process.env?.REACT_APP_ROS_WS_URL) ||
  "ws://localhost:8765";
const VIDEO_URL =
  (typeof process !== "undefined" && process.env?.REACT_APP_VIDEO_URL) ||
  "http://localhost:8080/stream?topic=/camera/image_raw";
const VIDEO_WIDTH = 1280;
const VIDEO_HEIGHT = 720;

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [currentPose, setCurrentPose] = useState("UNKNOWN");
  const [rawPose, setRawPose] = useState("UNKNOWN:0.00");
  const [confidence, setConfidence] = useState(0);
  const [landmarks, setLandmarks] = useState([]);

  useEffect(() => {
    const ros = new ROSLIB.Ros({ url: ROS_WS_URL });

    const handleOpen = () => setIsConnected(true);
    const handleClose = () => setIsConnected(false);
    const handleError = () => setIsConnected(false);

    ros.on("connection", handleOpen);
    ros.on("close", handleClose);
    ros.on("error", handleError);

    const currentPoseTopic = new ROSLIB.Topic({
      ros,
      name: "/pose_classifier/current_pose",
      messageType: "std_msgs/msg/String",
    });

    const rawPoseTopic = new ROSLIB.Topic({
      ros,
      name: "/pose_classifier/raw_pose",
      messageType: "std_msgs/msg/String",
    });

    const landmarksTopic = new ROSLIB.Topic({
      ros,
      name: "/mediapipe/pose_world_landmarks",
      messageType: "std_msgs/msg/Float32MultiArray",
    });

    currentPoseTopic.subscribe((message) => {
      setCurrentPose(message.data || "UNKNOWN");
    });

    rawPoseTopic.subscribe((message) => {
      const data = message.data || "UNKNOWN:0.00";
      setRawPose(data);

      const [label, confidenceText] = data.split(":");
      setCurrentPose(label || "UNKNOWN");

      const parsedConfidence = Number.parseFloat(confidenceText ?? "0");
      if (Number.isFinite(parsedConfidence)) {
        setConfidence(Math.max(0, Math.min(1, parsedConfidence)));
      } else {
        setConfidence(0);
      }
    });

    landmarksTopic.subscribe((message) => {
      setLandmarks(Array.isArray(message.data) ? message.data : []);
    });

    return () => {
      currentPoseTopic.unsubscribe();
      rawPoseTopic.unsubscribe();
      landmarksTopic.unsubscribe();
      ros.close();
    };
  }, []);

  return (
    <div
      style={{
        minHeight: "100vh",
        background:
          "linear-gradient(135deg, rgb(10, 18, 32) 0%, rgb(18, 38, 62) 55%, rgb(14, 72, 78) 100%)",
        color: "#f3f7fb",
        display: "flex",
        flexDirection: "column",
        fontFamily:
          '"SF Pro Display", "Segoe UI", "Helvetica Neue", sans-serif',
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "18px 24px",
          borderBottom: "1px solid rgba(255, 255, 255, 0.12)",
          backgroundColor: "rgba(8, 12, 20, 0.35)",
          backdropFilter: "blur(10px)",
        }}
      >
        <div style={{ fontSize: "1.2rem", fontWeight: 700 }}>
          Gesture Pose Interface
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "10px",
            fontSize: "0.95rem",
          }}
        >
          <span
            style={{
              width: "12px",
              height: "12px",
              borderRadius: "999px",
              backgroundColor: isConnected ? "#37d67a" : "#ff5d73",
              boxShadow: isConnected
                ? "0 0 14px rgba(55, 214, 122, 0.85)"
                : "0 0 14px rgba(255, 93, 115, 0.85)",
            }}
          />
          <span>{isConnected ? "Connected" : "Disconnected"}</span>
        </div>
      </div>

      <div
        style={{
          flex: 1,
          display: "grid",
          gridTemplateColumns: "minmax(320px, 1.1fr) minmax(280px, 0.9fr)",
          gap: "24px",
          padding: "24px",
        }}
      >
        <div
          style={{
            position: "relative",
            minHeight: "420px",
            borderRadius: "24px",
            overflow: "hidden",
            backgroundColor: "rgba(0, 0, 0, 0.28)",
            border: "1px solid rgba(255, 255, 255, 0.12)",
            boxShadow: "0 24px 60px rgba(0, 0, 0, 0.28)",
          }}
        >
          <img
            src={VIDEO_URL}
            alt="Camera feed"
            style={{
              width: "100%",
              height: "100%",
              display: "block",
              objectFit: "cover",
            }}
          />
          <SkeletonOverlay
            landmarks={landmarks}
            width={VIDEO_WIDTH}
            height={VIDEO_HEIGHT}
          />
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            gap: "22px",
            padding: "30px",
            borderRadius: "24px",
            backgroundColor: "rgba(5, 10, 18, 0.36)",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            boxShadow: "0 18px 40px rgba(0, 0, 0, 0.2)",
          }}
        >
          <div
            style={{
              fontSize: "0.95rem",
              textTransform: "uppercase",
              letterSpacing: "0.14em",
              color: "rgba(226, 235, 245, 0.72)",
            }}
          >
            Current Pose
          </div>

          <div
            style={{
              fontSize: "clamp(2.6rem, 5vw, 4.8rem)",
              lineHeight: 1,
              fontWeight: 800,
              textAlign: "center",
            }}
          >
            {currentPose}
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "0.95rem",
                color: "rgba(226, 235, 245, 0.82)",
              }}
            >
              <span>Confidence</span>
              <span>{Math.round(confidence * 100)}%</span>
            </div>
            <div
              style={{
                width: "100%",
                height: "16px",
                borderRadius: "999px",
                overflow: "hidden",
                backgroundColor: "rgba(255, 255, 255, 0.12)",
              }}
            >
              <div
                style={{
                  width: `${Math.round(confidence * 100)}%`,
                  height: "100%",
                  borderRadius: "999px",
                  background:
                    "linear-gradient(90deg, rgb(56, 189, 248) 0%, rgb(34, 197, 94) 100%)",
                  transition: "width 120ms ease-out",
                }}
              />
            </div>
          </div>

          <div
            style={{
              fontSize: "0.95rem",
              color: "rgba(226, 235, 245, 0.72)",
              padding: "16px 18px",
              borderRadius: "16px",
              backgroundColor: "rgba(255, 255, 255, 0.05)",
              wordBreak: "break-word",
            }}
          >
            Raw prediction: {rawPose}
          </div>
        </div>
      </div>
    </div>
  );
}
