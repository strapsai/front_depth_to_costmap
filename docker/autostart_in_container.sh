#!/bin/bash
source /opt/ros/humble/setup.bash
source /home/ros/workspace/install/setup.bash
ros2 launch depth_to_pointcloud_pub traversability_to_occupancygrid.launch.py