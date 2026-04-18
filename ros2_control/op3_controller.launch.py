import os
from pathlib import Path

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def _build_robot_description(xacro_path: Path, controller_config: Path) -> str:
    return xacro.process_file(
        str(xacro_path),
        mappings={"controller_config_file": str(controller_config)},
    ).toxml()


def generate_launch_description() -> LaunchDescription:
    os.environ["GAZEBO_MODEL_DATABASE_URI"] = ""
    os.environ["GAZEBO_MODEL_PATH"] = "/home/parv/ros2_ws/install/op3_description/share/op3_description/models"
    os.environ["GDK_BACKEND"] = "x11"
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ["XDG_SESSION_TYPE"] = "x11"
    os.environ["WAYLAND_DISPLAY"] = ""
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

    repo_root = Path(__file__).resolve().parents[1]
    world_file = repo_root / "simulation" / "worlds" / "demo_world.sdf"
    controller_config = repo_root / "ros2_control" / "forward_command_config.yaml"
    robot_xacro = repo_root / "ros2_control" / "op3_with_control.urdf.xacro"
    gazebo_launch = (
        Path(get_package_share_directory("gazebo_ros")) / "launch" / "gazebo.launch.py"
    )

    robot_description = _build_robot_description(robot_xacro, controller_config)

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True,
            }
        ],
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(gazebo_launch)),
        launch_arguments={"world": str(world_file), "gui": "false"}.items(),
    )

    spawn_op3 = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=[
            "-entity",
            "op3",
            "-topic",
            "robot_description",
            "-robot_namespace",
            "/",
            "-x",
            "0.0",
            "-y",
            "0.0",
            "-z",
            "0.35",
        ],
    )

    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
            "--param-file",
            str(controller_config),
        ],
    )

    forward_position_controller = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "forward_position_controller",
            "--controller-manager",
            "/controller_manager",
            "--param-file",
            str(controller_config),
        ],
    )

    action_dispatcher = ExecuteProcess(
        cmd=["python3", str(Path(__file__).with_name("action_dispatcher.py"))],
        output="screen",
    )

    foxglove_bridge = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        output="screen",
        parameters=[{"port": 8765, "use_sim_time": True}],
    )

    web_video_server = Node(
        package="web_video_server",
        executable="web_video_server",
        output="screen",
        parameters=[{"port": 8080, "use_sim_time": True}],
    )

    return LaunchDescription(
        [
            gazebo,
            robot_state_publisher,
            spawn_op3,
            TimerAction(period=1.0, actions=[joint_state_broadcaster]),
            TimerAction(period=2.0, actions=[forward_position_controller]),
            action_dispatcher,
            foxglove_bridge,
            web_video_server,
        ]
    )
