#!/bin/bash

roscore &
source /environment.sh
source /opt/ros/noetic/setup.bash
source /code/catkin_ws/devel/setup.bash --extend
source /code/submission_ws/devel/setup.bash --extend
source /code/solution/devel/setup.bash --extend

# set to 1 for a more verbose logging
export DEBUG=0

roslaunch --wait agent agent_node.launch &
roslaunch --wait car_interface all.launch veh:=$VEHICLE_NAME &
roslaunch --wait visual_lane_servoing visual_lane_servoing_node.launch veh:=$VEHICLE_NAME AIDO_eval:="true"
