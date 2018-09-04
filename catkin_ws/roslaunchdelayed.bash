#! /bin/bash
#ros_kill &
#sleep 5
export GAZEBO_MODEL_PATH=~/catkin_ws/src/sensor_stick/models
source ~/catkin_ws/devel/setup.bash

roslaunch sensor_stick robot_description.launch &
sleep 10
roslaunch sensor_stick robot_spawn.launch &
sleep 20
roslaunch sensor_stick robot_control.launch &





