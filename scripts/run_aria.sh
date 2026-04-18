#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-humble}"
MODE="auto"
USE_MOVEIT_IK="true"
SHOW_CAMERA="true"
INSTALL_SYSTEM="true"
START_MOVE_GROUP="false"
GAZEBO_GUI="true"
MOVEIT_PACKAGE="op3_moveit_config"
MOVEIT_LAUNCH_FILE="move_group.launch.py"

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --mode <auto|gazebo|nogazebo>   Pipeline mode (default: auto)
  --use-moveit-ik <true|false>    Enable MoveIt IK in retargeter (default: true)
  --show-camera <true|false>      Show MediaPipe preview window (default: true)
  --install-system <true|false>   Install ROS/system deps via apt (default: true)
  --start-move-group <true|false> Start move_group separately (default: false)
  --gazebo-gui <true|false>       Launch Gazebo client GUI in gazebo mode (default: true)
  --moveit-package <name>         MoveIt config package (default: op3_moveit_config)
  --moveit-launch-file <file>     MoveIt launch file (default: move_group.launch.py)
  -h, --help                      Show this help

Examples:
  ./scripts/run_aria.sh --mode nogazebo
  ./scripts/run_aria.sh --mode gazebo --use-moveit-ik true
  ./scripts/run_aria.sh --mode nogazebo --start-move-group true --moveit-package my_op3_moveit_config
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --use-moveit-ik)
      USE_MOVEIT_IK="$2"
      shift 2
      ;;
    --show-camera)
      SHOW_CAMERA="$2"
      shift 2
      ;;
    --install-system)
      INSTALL_SYSTEM="$2"
      shift 2
      ;;
    --start-move-group)
      START_MOVE_GROUP="$2"
      shift 2
      ;;
    --gazebo-gui)
      GAZEBO_GUI="$2"
      shift 2
      ;;
    --moveit-package)
      MOVEIT_PACKAGE="$2"
      shift 2
      ;;
    --moveit-launch-file)
      MOVEIT_LAUNCH_FILE="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

cleanup() {
  pkill -f "op3_controller.launch.py|op3_nogazebo_pipeline.launch.py|moveit_ik_retargeting_node.py|action_dispatcher.py|mediapipe_node.py|ros2 launch ${MOVEIT_PACKAGE}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

safe_source() {
  local script_path="$1"
  set +u
  # shellcheck disable=SC1090
  source "$script_path"
  set -u
}

wait_for_service() {
  local service_name="$1"
  local timeout_sec="$2"
  local elapsed=0

  while (( elapsed < timeout_sec )); do
    if ros2 service list 2>/dev/null | grep -Fxq "$service_name"; then
      return 0
    fi
    sleep 1
    ((elapsed += 1))
  done
  return 1
}

safe_source "/opt/ros/${ROS_DISTRO}/setup.bash"
if [[ -f "/home/${USER}/ros2_ws/install/setup.bash" ]]; then
  safe_source "/home/${USER}/ros2_ws/install/setup.bash"
fi

if [[ "${INSTALL_SYSTEM}" == "true" ]]; then
  if command -v sudo >/dev/null 2>&1; then
    echo "[setup] Installing system dependencies (requires sudo)..."
    sudo apt-get update || true
    sudo apt-get install -y \
      python3-venv \
      python3-pip \
      python3-lxml \
      ros-${ROS_DISTRO}-xacro \
      ros-${ROS_DISTRO}-gazebo-ros \
      ros-${ROS_DISTRO}-gazebo-ros2-control \
      ros-${ROS_DISTRO}-controller-manager \
      ros-${ROS_DISTRO}-forward-command-controller \
      ros-${ROS_DISTRO}-joint-state-broadcaster \
      ros-${ROS_DISTRO}-joint-trajectory-controller \
      ros-${ROS_DISTRO}-robot-state-publisher \
      ros-${ROS_DISTRO}-moveit \
      ros-${ROS_DISTRO}-moveit-ros-move-group \
      ros-${ROS_DISTRO}-foxglove-bridge \
      ros-${ROS_DISTRO}-web-video-server || true
  else
    echo "[setup] sudo not found; skipping apt installs"
  fi
fi

VENV_PATH="${REPO_ROOT}/.venv"
if [[ ! -d "${VENV_PATH}" ]]; then
  echo "[setup] Creating virtual environment at ${VENV_PATH}"
  python3 -m venv "${VENV_PATH}"
fi

safe_source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel >/dev/null
python -m pip install -r "${REPO_ROOT}/perception/requirements.txt"

if [[ "${MODE}" == "auto" ]]; then
  if ros2 pkg prefix gazebo_ros >/dev/null 2>&1; then
    MODE="gazebo"
  else
    MODE="nogazebo"
  fi
fi

if [[ "${MODE}" == "gazebo" ]] && [[ "${GAZEBO_GUI}" == "true" ]] && [[ -z "${DISPLAY:-}" ]]; then
  echo "[warn] DISPLAY is not set; forcing --gazebo-gui false (headless session)"
  GAZEBO_GUI="false"
fi

if [[ "${START_MOVE_GROUP}" == "true" ]]; then
  if ros2 pkg prefix "${MOVEIT_PACKAGE}" >/dev/null 2>&1; then
    echo "[run] Starting move_group from ${MOVEIT_PACKAGE}/${MOVEIT_LAUNCH_FILE}"
    ros2 launch "${MOVEIT_PACKAGE}" "${MOVEIT_LAUNCH_FILE}" &
    sleep 2
  else
    echo "[warn] MoveIt package '${MOVEIT_PACKAGE}' not found; move_group not started"
  fi
fi

if [[ "${MODE}" == "gazebo" ]]; then
  if ! ros2 pkg prefix gazebo_ros >/dev/null 2>&1; then
    echo "[error] gazebo_ros package not found. Install ROS Gazebo packages or run with --mode nogazebo"
    exit 1
  fi

  echo "[run] Starting Gazebo control pipeline"
  ros2 launch "${REPO_ROOT}/ros2_control/op3_controller.launch.py" "gazebo_gui:=${GAZEBO_GUI}" &

  echo "[run] Waiting for controller manager service"
  if ! wait_for_service "/controller_manager/list_controllers" 25; then
    echo "[error] /controller_manager/list_controllers did not appear within timeout"
    echo "[hint] Common cause: missing python3-lxml breaks spawn_entity.py"
    echo "[hint] Try: sudo apt-get install -y python3-lxml ros-${ROS_DISTRO}-gazebo-ros2-control"
    exit 1
  fi

  echo "[run] Starting retargeting node (use_ik=${USE_MOVEIT_IK})"
  python3 "${REPO_ROOT}/retargeting/moveit_ik_retargeting_node.py" --ros-args -p "use_ik:=${USE_MOVEIT_IK}" &
  sleep 1
else
  echo "[run] Starting no-Gazebo control pipeline (use_ik=${USE_MOVEIT_IK})"
  ros2 launch "${REPO_ROOT}/ros2_control/op3_nogazebo_pipeline.launch.py" \
    "launch_mock_input:=false" \
    "use_moveit_ik:=${USE_MOVEIT_IK}" \
    "launch_move_group:=false" &
  sleep 2
fi

echo "[run] Starting MediaPipe perception (show_debug_window=${SHOW_CAMERA})"
python3 "${REPO_ROOT}/perception/mediapipe_node.py" --ros-args -p "show_debug_window:=${SHOW_CAMERA}" &

echo "[ok] ARIA pipeline started in '${MODE}' mode"
echo "[ok] Key topics: /mediapipe/pose_world_landmarks /op3/joint_commands /forward_position_controller/commands"
echo "[hint] Press Ctrl+C to stop all started processes"

wait
