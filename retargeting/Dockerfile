FROM ros:humble-ros-base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy pyyaml

COPY . /app

CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && python3 retargeting_node.py"]
