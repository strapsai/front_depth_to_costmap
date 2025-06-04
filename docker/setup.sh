#!/bin/bash
cd /home/ros/workspace/
vcs import < /home/ros/workspace/src/elevation_mapping_cupy/docker/src.repos src/ --recursive -w $(($(nproc)/2))

sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -y -r --rosdistro humble --skip-keys="gazebo_ros_pkgs ros-humble-gazebo-ros-pkgs"






# rosdep update
# sudo rosdep init
# echo 'rosdep init'
# sudo apt update
# sudo apt install -y python3-rosdep
# rosdep install --from-paths src --ignore-src -y -r
