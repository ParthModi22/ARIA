"""Launch OP3 retargeting/control pipeline without Gazebo.

This launch file keeps the topic contracts intact while avoiding simulator startup.
It can optionally start MoveIt `move_group` if an OP3 MoveIt config package exists.
"""

from __future__ import annotations

from pathlib import Path

import xacro
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _build_robot_description(xacro_path: Path, controller_config: Path) -> str:
    return xacro.process_file(
        str(xacro_path),
        mappings={"controller_config_file": str(controller_config)},
    ).toxml()


def generate_launch_description() -> LaunchDescription:
    repo_root = Path(__file__).resolve().parents[1]
    controller_config = repo_root / "ros2_control" / "forward_command_config.yaml"
    robot_xacro = repo_root / "ros2_control" / "op3_with_control.urdf.xacro"

    robot_description = _build_robot_description(robot_xacro, controller_config)

    launch_move_group_arg = DeclareLaunchArgument(
        "launch_move_group",
        default_value="false",
        description="Launch `ros2 launch <moveit_package> <moveit_launch_file>`.",
    )
    moveit_package_arg = DeclareLaunchArgument(
        "moveit_package",
        default_value="op3_moveit_config",
        description="MoveIt config package name.",
    )
    moveit_launch_file_arg = DeclareLaunchArgument(
        "moveit_launch_file",
        default_value="move_group.launch.py",
        description="MoveIt launch file inside moveit_package.",
    )
    launch_mock_input_arg = DeclareLaunchArgument(
        "launch_mock_input",
        default_value="true",
        description="Launch deterministic mock MediaPipe publisher.",
    )
    use_moveit_ik_arg = DeclareLaunchArgument(
        "use_moveit_ik",
        default_value="true",
        description="Enable MoveIt IK calls in retargeting node.",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": False,
            }
        ],
    )

    move_group = ExecuteProcess(
        cmd=[
            "ros2",
            "launch",
            LaunchConfiguration("moveit_package"),
            LaunchConfiguration("moveit_launch_file"),
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("launch_move_group")),
    )

    action_dispatcher = ExecuteProcess(
        cmd=["python3", str(repo_root / "ros2_control" / "action_dispatcher.py")],
        output="screen",
    )

    mock_input = ExecuteProcess(
        cmd=["python3", str(repo_root / "tests" / "tools" / "mock_mediapipe_publisher.py")],
        output="screen",
        condition=IfCondition(LaunchConfiguration("launch_mock_input")),
    )

    # Retargeting runtime parameters are injected through ROS args.
    retargeting_node_with_args = ExecuteProcess(
        cmd=[
            "python3",
            str(repo_root / "retargeting" / "moveit_ik_retargeting_node.py"),
            "--ros-args",
            "-p",
            ["use_ik:=", LaunchConfiguration("use_moveit_ik")],
            "-p",
            "publish_rate_hz:=30.0",
            "-p",
            "ik_timeout_sec:=0.006",
        ],
        output="screen",
        additional_env={
            "RCUTILS_LOGGING_BUFFERED_STREAM": "1",
        },
    )

    return LaunchDescription(
        [
            launch_move_group_arg,
            moveit_package_arg,
            moveit_launch_file_arg,
            launch_mock_input_arg,
            use_moveit_ik_arg,
            robot_state_publisher,
            move_group,
            action_dispatcher,
            mock_input,
            retargeting_node_with_args,
        ]
    )
