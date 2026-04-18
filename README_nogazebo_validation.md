# OP3 Full-Body Pipeline Validation (No-Gazebo + Gazebo Final)

This runbook validates full-body OP3 retargeting quickly without Gazebo, then shows the final Gazebo execution path.

## 0) One-command setup + run (both Linux PCs)

Use the portable script below. It installs dependencies, prepares a local Python venv, and runs the right pipeline.

```bash
cd /home/arya/Desktop/MOcap/ARIA
chmod +x ./scripts/run_aria.sh
```

No-Gazebo machine (like your current PC):

```bash
./scripts/run_aria.sh --mode nogazebo --use-moveit-ik true --show-camera true
```

Gazebo-capable machine:

```bash
./scripts/run_aria.sh --mode gazebo --use-moveit-ik true --show-camera true
```

Auto-detect mode (chooses Gazebo when available):

```bash
./scripts/run_aria.sh --mode auto --use-moveit-ik true
```

Notes:
- The script will try to install apt packages when `sudo` is available.
- For MoveIt compute IK service, optionally add `--start-move-group true --moveit-package <your_moveit_config_pkg>`.

## What was added

- `retargeting/moveit_ik_retargeting_node.py`
- `ros2_control/op3_nogazebo_pipeline.launch.py`
- `tests/tools/mock_mediapipe_publisher.py`
- `tests/tools/pipeline_verifier.py`
- `tests/tools/joint_command_observer.py`
- `tests/launch/ik_pipeline_nogazebo.launch.py`

## 1) No-Gazebo quick validation (recommended first)

### Fallback mode (no MoveIt server required)

```bash
source /opt/ros/humble/setup.bash
source /home/arya/ros2_ws/install/setup.bash
ros2 launch /home/arya/Desktop/MOcap/ARIA/tests/launch/ik_pipeline_nogazebo.launch.py use_moveit_ik:=false launch_move_group:=false
```

Expected:
- `/op3/joint_commands` publishes 20-joint `JointState`
- `/forward_position_controller/commands` publishes 20-element `Float64MultiArray`
- `pipeline_verifier.py` reports PASS on all checks

### MoveIt IK mode (if MoveIt config exists)

If your MoveIt config exists (e.g. `op3_moveit_config`), run:

```bash
source /opt/ros/humble/setup.bash
source /home/arya/ros2_ws/install/setup.bash
ros2 launch /home/arya/Desktop/MOcap/ARIA/tests/launch/ik_pipeline_nogazebo.launch.py use_moveit_ik:=true launch_move_group:=true
```

If your package name/launch file differs:

```bash
source /opt/ros/humble/setup.bash
source /home/arya/ros2_ws/install/setup.bash
ros2 launch /home/arya/Desktop/MOcap/ARIA/ros2_control/op3_nogazebo_pipeline.launch.py launch_move_group:=true moveit_package:=<your_moveit_pkg> moveit_launch_file:=<your_move_group_launch.py>
```

## 2) See measurable full-body results without Gazebo

Run the observer in a second terminal while the no-Gazebo pipeline is running:

```bash
source /opt/ros/humble/setup.bash
source /home/arya/ros2_ws/install/setup.bash
python3 /home/arya/Desktop/MOcap/ARIA/tests/tools/joint_command_observer.py --ros-args -p duration_sec:=10.0 -p min_messages:=40
```

Observer output includes:
- per-joint min/max/amplitude/rms
- estimated publish rate
- moving-joint count (`amp > 0.03 rad`)

For complete-body fallback mode, you should see motion in arms, legs, ankles, and head, with hip yaw changing during torso twist.

## 2.1) See MediaPipe camera preview window

If you want the camera window on screen (desktop session), run:

```bash
source /opt/ros/humble/setup.bash
source /home/arya/ros2_ws/install/setup.bash
/home/arya/Desktop/MOcap/.venv/bin/python /home/arya/Desktop/MOcap/ARIA/perception/mediapipe_node.py --ros-args -p show_debug_window:=true
```

If you are running headless/no display, disable preview explicitly:

```bash
source /opt/ros/humble/setup.bash
source /home/arya/ros2_ws/install/setup.bash
/home/arya/Desktop/MOcap/.venv/bin/python /home/arya/Desktop/MOcap/ARIA/perception/mediapipe_node.py --ros-args -p show_debug_window:=false
```

## 3) Final execution in Gazebo

When you are ready for the full simulation execution path:

Terminal A (Gazebo + controllers + dispatcher):

```bash
source /opt/ros/humble/setup.bash
source /home/arya/ros2_ws/install/setup.bash
ros2 launch /home/arya/Desktop/MOcap/ARIA/ros2_control/op3_controller.launch.py
```

Terminal B (perception + retargeting):

```bash
source /opt/ros/humble/setup.bash
source /home/arya/ros2_ws/install/setup.bash
ros2 launch /home/arya/Desktop/MOcap/ARIA/perception_pipeline.launch.py
```

If you are using the MoveIt IK retargeter script directly, run it instead of the legacy retargeting node:

```bash
source /opt/ros/humble/setup.bash
source /home/arya/ros2_ws/install/setup.bash
python3 /home/arya/Desktop/MOcap/ARIA/retargeting/moveit_ik_retargeting_node.py --ros-args -p use_ik:=true
```

## 4) Useful checks (both modes)

```bash
ros2 topic list | grep -E '/mediapipe/pose_world_landmarks|/op3/joint_commands|/forward_position_controller/commands'
ros2 topic hz /op3/joint_commands
ros2 topic hz /forward_position_controller/commands
ros2 topic echo /op3/joint_commands --once
ros2 topic echo /forward_position_controller/commands --once
```

## 5) Interpreting subscriber count and joint angles

If you run no-Gazebo mode and see subscriber count `0` on `/forward_position_controller/commands`, that can be expected.

- `/op3/joint_commands` contains commanded OP3 joint angles (`sensor_msgs/JointState`) in radians.
- `action_dispatcher.py` republishes those angles to `/forward_position_controller/commands` as ordered numeric arrays.
- In **no-Gazebo mode**, there is no `controller_manager` + `forward_position_controller` process by default, so published commands may have no downstream subscriber.
- In **Gazebo mode** (`op3_controller.launch.py`), controller spawners are launched, so `/forward_position_controller/commands` should have active subscribers and drive simulation.

Check publisher/subscriber counts explicitly:

```bash
source /opt/ros/humble/setup.bash
source /home/arya/ros2_ws/install/setup.bash
ros2 topic info -v /op3/joint_commands
ros2 topic info -v /forward_position_controller/commands
```

To verify command validity for OP3, inspect one message:

```bash
source /opt/ros/humble/setup.bash
source /home/arya/ros2_ws/install/setup.bash
ros2 topic echo /op3/joint_commands --once
```

The `name` field should match OP3 controller order (`l_sho_pitch ... head_tilt`), and `position` values are the real-time command angles being generated by the retargeter.
