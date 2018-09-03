#!/usr/bin/env python

# Import modules
from mpmath import *
from sympy import *
import numpy
import math
import os
import sys
import pickle
import collections
import types
import traceback
#import pprint
#pp = pprint.PrettyPrinter(indent=4)

import rospy
import tf
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
import sensor_msgs.point_cloud2 as pc2

import pcl_helper

#====================== GLOBALS =====================
g_pcl_sub = None
g_pcl_objects_pub = None
g_pcl_table_pub = None
g_callBackCount = 0

#--------------------------------- CB_PCL()
def CB_PCL(msgPCL):
    global g_callBackCount
    g_callBackCount += 1
    if (g_callBackCount % 10 != 0):
        return;

    print "\rCBCount= {:05d}".format(g_callBackCount),
    sys.stdout.flush()

    #if (g_callBackCount == 10):
    #    fileNameOut = "msgPCL.pypickle"
    #    pickle.dump(msgPCL, open(fileNameOut, "wb"))

    pclRecs = [] # For dev/debug display. Container for point cloud records: tuple (pclObj, pclName)

    # TODO: Convert ROS msg to PCL data
    pclpcIn = pcl_helper.ros_to_pcl(msgPCL)
    pclRecs.append((pclpcIn, "pclpcIn"))

    # Create a VoxelGrid filter object for our input point cloud
    vox = pclpcIn.make_voxel_grid_filter()
    voxelSize = 0.01
    vox.set_leaf_size(voxelSize, voxelSize, voxelSize)

    # TODO: Voxel Grid Downsampling
    # Call the filter function to obtain the resultant downsampled point cloud
    pclpcVoxels = vox.filter()
    pclRecs.append((pclpcVoxels, "pclpcDownSampled"))

    # TODO: PassThrough Filter


    # TODO: Extract inliers and outliers

    # TODO: Euclidean Clustering

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately

    # TODO: Convert PCL data to ROS messages

    # TODO: Publish ROS messages
    g_pcl_objects_pub.publish(msgPCL)
    g_pcl_table_pub.publish(msgPCL)


#====================== Main() =====================
def Main():
    print("Ros.clustering node init...")

    global g_pcl_sub
    global g_pcl_objects_pub
    global g_pcl_table_pub

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    g_pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, CB_PCL, queue_size=1)

    # TODO: Create Publishers
    g_pcl_objects_pub = rospy.Publisher("/pcl_objects", pcl_helper.PointCloud2, queue_size=1)
    g_pcl_table_pub = rospy.Publisher("/pcl_table", pcl_helper.PointCloud2, queue_size=1)

    # Initialize color_list
    pcl_helper.get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        print("Ros.clustering node statrted.")
        rospy.spin()


#====================== Invoke Main() =====================
if __name__ == '__main__':
    Main()

# Auto Invocation of Main()
Main()