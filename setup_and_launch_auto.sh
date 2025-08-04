#!/bin/bash
set -e

echo "[INFO] Starting smart setup..."

# === [1] Check & download model ===
MODEL_PATH="traversability_model/traversability_model.plan"
MODEL_URL="https://www.dropbox.com/scl/fi/xxjtu4hzdb5f8qwu27ack/traversability_model.plan?rlkey=8n7udgy6l8vlt3sm3fo57odiy&st=aukug1k4&dl=1"

if [ ! -f "$MODEL_PATH" ]; then
  echo "[INFO] Model file not found. Downloading..."
  wget -O "$MODEL_PATH" "$MODEL_URL"
else
  echo "[INFO] Model file already exists: $MODEL_PATH"
fi

# === [2] Check Docker image ===
IMAGE_NAME="theairlab/dtc:jp6.1-01-torch"
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
  echo "[INFO] Docker image not found. Pulling..."
  docker pull $IMAGE_NAME
else
  echo "[INFO] Docker image already exists: $IMAGE_NAME"
fi

# === [3] Run container if needed ===
CONTAINER_NAME="front_traversability"
WORKDIR_HOST=$(pwd)
CONTAINER_WORKDIR="/home/ros/workspace/src/front_depth_to_costmap"


if [[ "$(docker ps -aq -f name=$CONTAINER_NAME)" != "" ]]; then
  if [[ "$(docker ps -q -f name=$CONTAINER_NAME)" == "" ]]; then
    echo "[INFO] Container exists but not running. Starting..."
    docker start $CONTAINER_NAME
  fi

  echo "[INFO] Checking if container mount is still valid..."
  if ! docker exec $CONTAINER_NAME test -d "$CONTAINER_WORKDIR"; then
    echo "[WARN] Mount path invalid inside container. Removing and recreating container..."
    docker rm -f $CONTAINER_NAME
  else
    echo "[INFO] Container is already running. Mount valid."
  fi
fi

if [[ "$(docker ps -aq -f name=$CONTAINER_NAME)" == "" ]]; then
  echo "[INFO] Container does not exist. Running for the first time..."
  mkdir -p .etc && pushd .etc > /dev/null
  ln -sf /etc/passwd .
  ln -sf /etc/shadow .
  ln -sf /etc/group .
  popd > /dev/null


  RUN_COMMAND="docker run \
    --name $CONTAINER_NAME \
    --ulimit rtprio=99 \
    --cap-add=sys_nice \
    --privileged \
    --net=host \
    --ipc=host \
    --runtime=nvidia \
    -e HOST_USERNAME=$(whoami) \
    -v $WORKDIR_HOST:$CONTAINER_WORKDIR \
    -w $CONTAINER_WORKDIR \
    -d $IMAGE_NAME tail -f /dev/null"

  echo -e "[setup_and_launch_auto.sh]: \e[1;32mRunning container...\n\e[0;35m$RUN_COMMAND\e[0m"
  eval $RUN_COMMAND
fi

echo "[INFO] Waiting for container to start..."
while [[ "$(docker inspect -f '{{.State.Running}}' $CONTAINER_NAME 2>/dev/null)" != "true" ]]; do
  sleep 1
done


# === [4] Run setup + build + launch inside container ===
docker exec -it $CONTAINER_NAME bash -i -c "
  set -e
  if ! command -v ros2 &> /dev/null; then
    echo '[INFO] ROS 2 not found. Installing ROS 2 Humble...'

    apt update && apt install -y curl gnupg lsb-release locales

    locale-gen en_US en_US.UTF-8
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
    export LANG=en_US.UTF-8

    rm -f /usr/share/keyrings/ros-archive-keyring.gpg
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | \
      gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

    echo 'deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main' > /etc/apt/sources.list.d/ros2.list

    apt update
    apt install -y ros-humble-desktop python3-colcon-common-extensions

    echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc

    echo '[INFO] ROS 2 Humble installed.'
  else
    echo '[INFO] ROS 2 already installed.'
  fi

  cd /home/ros/workspace
  source /opt/ros/humble/setup.bash
  if [ -f install/setup.bash ]; then
    source install/setup.bash
  fi

  if ! ros2 launch depth_to_pointcloud_pub traversability_to_occupancygrid.launch.py; then
    echo '[WARN] Rebuilding...'
    rm -rf build install log
    colcon build --symlink-install
    source install/setup.bash
    echo '[INFO] Retrying...'
    ros2 launch depth_to_pointcloud_pub traversability_to_occupancygrid.launch.py
  fi
"

