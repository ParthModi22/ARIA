"""Integration launch: no-Gazebo OP3 IK pipeline + optional verifier."""

from __future__ import annotations

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    repo_root = Path(__file__).resolve().parents[2]

    launch_move_group_arg = DeclareLaunchArgument(
        "launch_move_group",
        default_value="false",
        description="Start MoveIt move_group if moveit config exists.",
    )
    use_moveit_ik_arg = DeclareLaunchArgument(
        "use_moveit_ik",
        default_value="true",
        description="Enable MoveIt IK calls in retargeter.",
    )
    run_verifier_arg = DeclareLaunchArgument(
        "run_verifier",
        default_value="true",
        description="Run pipeline verifier after startup.",
    )

    core_pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(repo_root / "ros2_control" / "op3_nogazebo_pipeline.launch.py")
        ),
        launch_arguments={
            "launch_move_group": LaunchConfiguration("launch_move_group"),
            "use_moveit_ik": LaunchConfiguration("use_moveit_ik"),
            "launch_mock_input": "true",
        }.items(),
    )

    verifier = ExecuteProcess(
        cmd=["python3", str(repo_root / "tests" / "tools" / "pipeline_verifier.py")],
        output="screen",
        condition=IfCondition(LaunchConfiguration("run_verifier")),
    )

    return LaunchDescription(
        [
            launch_move_group_arg,
            use_moveit_ik_arg,
            run_verifier_arg,
            core_pipeline,
            TimerAction(period=3.0, actions=[verifier]),
        ]
    )
