"""Launch the perception pipeline: mediapipe → retargeting.

The classifier is currently disabled so movement development can proceed without
the classifier process interfering.  Re-enable pose_classifier below once
gesture classification is needed.
"""

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
        cmd=["python3", str(base / "retargeting" / "retargeting_node.py")],
        output="screen",
    )

    return LaunchDescription([
        mediapipe_node,
        # pose_classifier,  # disabled — enable when classifier is needed
        retargeting_node,
    ])
