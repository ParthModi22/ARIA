FROM ros:humble-ros-base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir mediapipe==0.10.21 numpy

COPY . /app

CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && python3 mediapipe_node.py"]
