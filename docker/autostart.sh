#!/bin/bash

docker start front_traversability;

sleep 1;

docker exec front_traversability bash /home/ros/workspace/src/front_depth_to_costmap/docker/autostart_in_container.sh;

exit 0;
