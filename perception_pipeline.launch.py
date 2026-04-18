"""Launch the full perception pipeline: mediapipe → classifier → full-body retargeting."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description() -> LaunchDescription:
    base = Path(__file__).resolve().parent

    mediapipe_node = ExecuteProcess(
        cmd=["python3", str(base / "perception" / "mediapipe_node.py")],
        output="screen",
    )

    pose_classifier = ExecuteProcess(
        cmd=["python3", str(base / "classifier" / "pose_classifier.py")],
        output="screen",
    )

    retargeting_node = ExecuteProcess(
        cmd=[
            "python3",
            str(base / "retargeting" / "moveit_ik_retargeting_node.py"),
            "--ros-args",
            "-p",
            "use_ik:=true",
        ],
        output="screen",
    )

    return LaunchDescription([
        mediapipe_node,
        pose_classifier,
        retargeting_node,
    ])
