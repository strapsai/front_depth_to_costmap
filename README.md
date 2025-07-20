# ROS 2 costmap generated from image-based traversability  (for AGX ORIN)
## Installation

You will need to install the following:
- JetPack 6.2
- Docker
- NVIDIA Container Toolkit
- NVIDIA CUDA Toolkit
- Dockerhub ID : alsgh000118@naver.com
- Username : alsgh000118
- Dockerhub password : alsgh001!


## Setup

### (1) Clone the Repository

```bash
(local)$ git clone https://github.com/strapsai/front_depth_to_costmap.git
```



### (2) Build the Docker Container
```bash
(local)$ docker login -u <Username> -p <Password> docker.io
```

```bash
(local)$ cd depth_to_elevation_map/docker
(local)$ docker pull alsgh000118/rcv-dtc:0.51
(local)$ ./run_mhlee-rcv-dtc-0.5.sh
```

### (3) Download the inference model file
```bash
(local)$ docker에 inference model다운받는 방법 추가하기
```


### (4) In Docker command shell, Check the ORIN GPU Conditions
```bash
(local)$ sudo docker cp /usr/bin/tegrastats <container ID>:/usr/bin/tegrastats
(docker)$ tegrastats | grep gpu
```


### (5) Setup the Workspace

```bash
(docker)$ cd docker
(docker)$ ./setup.sh
```


### (6) Build the Workspace

```bash
(docker)$ cd /home/ros/workspace/
(docker)$ source /opt/ros/humble/setup.bash
(docker)$ colcon build --symlink-install
(docker)$ source install/setup.bash
(docker)$ source /opt/ros/humble/setup.bash

 If you see a build error like "could not find package `ament_cmake`", run `source /opt/ros/humble/setup.bash` and try building again.
```




### (7) 컨테이너 실행하면 런치파일 실행 가능하도록 명령어 구성해야 할듯


## Running the Demo



### TERMINAL : Launch the `traversability_to_occupancygrid` node inside the container

Convert frontleft, frontright depth data into merged point cloud.

Run `setup.bash` only on the first try.

```bash
(local)$ docker exec -it front_depth_costmap bash
(docker)$ cd /home/ros/workspace/
(docker)$ source install/setup.bash 
(docker)$ source /opt/ros/humble/setup.bash
(docker)$ export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
(docker)$ ros2 launch depth_to_pointcloud_pub traversability_to_occupancygrid.launch.py


```


<details>
<summary><strong>backup</strong></summary>

<br>


# ROS 2 costmap generated from image-based traversability  (for AGX ORIN)

**Status**: 🚧 Under Development  
---

## Installation

You will need to install the following:
- JetPack 6.2
- Docker
- NVIDIA Container Toolkit
- NVIDIA CUDA Toolkit
- Dockerhub ID : alsgh000118@naver.com
- Username : alsgh000118
- Dockerhub password : alsgh001!

## Setup

### (1) Clone the Repository

```bash
(local)$ git clone https://github.com/strapsai/front_depth_to_costmap.git
```



### (2) Build the Docker Container
```bash
(local)$ docker login -u <Username> -p <Password> docker.io
```

```bash
(local)$ cd depth_to_elevation_map/docker
(local)$ docker pull alsgh000118/rcv-dtc:0.51
(local)$ ./run_mhlee-rcv-dtc-0.5.sh
```

### (3) In Docker command shell, Check the ORIN GPU Conditions
```bash
(local)$ sudo docker cp /usr/bin/tegrastats <container ID>:/usr/bin/tegrastats
(docker)$ tegrastats | grep gpu
```


### (4) Setup the Workspace

```bash
(docker)$ cd docker
(docker)$ ./setup.sh
```

<details>
<summary><strong>(4-1) Errors During Installation in Docker Environment</strong></summary>

(docker)

- If simple-parsing install error (Already applied, May. 29. 2025)
  ```bash
  ERROR: Cannot locate rosdep definition for simple-parsing
  This error is caused by [ Filename:, Line: ]
  This code was commented.
  ```

- ament_python install error (Already applied, May. 29. 2025)
  ```bash
  ERROR: Cannot locate rosdep definition for ament_python
  The error was resolved by commenting out ament_python in the package.xml of the depth_to_pointcloud_pub package.
  This error is caused by [ Filename: depth_to_elevation_map/depth_to_pointcloud_pub/package.xml, Line: 10~12 ]
  This code was commented.
  ```

- Gazebo package install error (Already applied, May. 29. 2025)
  ```bash
  This error can be resolved by modifying setup.sh as follows:
  rosdep install --from-paths src --ignore-src -y -r --rosdistro humble --skip-keys="gazebo_ros_pkgs ros-humble-gazebo-ros-pkgs"
  ```

</details>


### (5) Build the Workspace

```bash
(docker)$ cd /home/ros/workspace/
(docker)$ source /opt/ros/humble/setup.bash
(docker)$ colcon build --symlink-install
(docker)$ source install/setup.bash
(docker)$ source /opt/ros/humble/setup.bash

 If you see a build error like "could not find package `ament_cmake`", run `source /opt/ros/humble/setup.bash` and try building again.
```



## Running the Demo


### TERMINAL 1. Play the `.mcap` data in Local or Docker
<s> docker cp <host_path> elevation_mapping_cupy:/home/ros/workspace/src/elevation_mapping_cupy/ </s>

Before run the .mcap file, you should install following package.
```bash
(local or docker)$ sudo apt update
(local or docker)$ sudo apt install ros-humble-rosbag2-storage-mcap

# if you use [local], you should install following package.
(local)$ sudo apt update
(local)$ sudo apt install ros-humble-rmw-cyclonedds-cpp

(local or docker)$ export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
(local or docker)$ source /opt/ros/humble/setup.bash
(local or docker)$ ros2 bag play data.mcap
```

### TERMINAL 2. Launch the `elevation_mapping_cupy` node

In the terminal where you built the workspace, 
```bash
(docker)$ export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
(docker)$ ros2 launch elevation_mapping_cupy elevation_mapping_cupy.launch.py
```

### TERMINAL 3. Launch the `traversability_to_occupancygrid` node inside the container

Convert frontleft, frontright depth data into merged point cloud.

Run `setup.bash` only on the first try.

```bash
(local)$ docker exec -it front_depth_costmap bash
(docker)$ cd /home/ros/workspace/
(docker)$ source install/setup.bash 
(docker)$ source /opt/ros/humble/setup.bash
(docker)$ export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
(docker)$ ros2 launch depth_to_pointcloud_pub traversability_to_occupancygrid.launch.py


```




### (Option) (Fourth terminal) play the 'image'

To play the image synchronized with the rosbag, run:

```bash
docker exec -it elevation_mapping_cupy bash
python3 image_play.py
```


## Elevation Mapping Cupy Setting

### 1. Single Pointcloud Mode

To enable single pointcloud mode, set the following parameter to true.

To accumulate pointclouds over time, set it to false instead:
```yaml
clear_map_before_update: true # or false
```

in the configuration file:

```
/home/ros/workspace/src/elevation_mapping_cupy/elevation_mapping_cupy/config/core/core_param.yaml
```


## ⚠️ Notes
### 1. If "ros2 topic list" shows the topics but "ros2 topic echo <topic/name>" prints nothing (possibly due to communication issues)
- In the author's case, this was resolved by, 
```
(local) sudo apt update
(local) sudo apt install ros-humble-rmw-cyclonedds-cpp
(local or docker) export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```
You can replace rmw_cyclonedds_cpp with another implementation like rmw_fastrtps_cpp, depending on your environment.

### 2. If gridmap is not available in the local RViz:
Install the grid map plugin with:

```
(local) sudo apt update
(local) sudo apt install ros-humble-grid-map-rviz-plugin
```

### 3. XAUTH Configuration Commented Out

During `docker run`, XAUTH-related errors frequently occurred,  
so the corresponding configuration was commented out in the script.

If you plan to use GUI tools such as RViz in the future,  
I recommend to **uncomment** the following section and try again:

```bash
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    touch $XAUTH
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
    chmod a+r $XAUTH
fi
```

Also, make sure to **add the following options** to the `RUN_COMMAND`:

```bash
# If using GUI tools, add the following:
  --volume=$XSOCK:$XSOCK:rw \
  --volume=$XAUTH:$XAUTH:rw \
  --env="QT_X11_NO_MITSHM=1" \
  --env="XAUTHORITY=$XAUTH" \
  --env="DISPLAY=$DISPLAY" \

</details>
