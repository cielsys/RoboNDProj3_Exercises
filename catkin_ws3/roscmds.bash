#! /bin/bash

# Capture and training
roslaunch sensor_stick training.launch &
rosrun sensor_stick capture_features.py &
rosrun sensor_stick train_svm.py &

# The real thing
roslaunch sensor_stick robot_spawn.launch
