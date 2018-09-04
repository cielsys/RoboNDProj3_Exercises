#! /bin/bash
#ros_kill &
#sleep 5
roslaunch sensor_stick robot_description.launch &
sleep 10
roslaunch sensor_stick robot_spawn.launch &
sleep 20
roslaunch sensor_stick robot_control.launch &




