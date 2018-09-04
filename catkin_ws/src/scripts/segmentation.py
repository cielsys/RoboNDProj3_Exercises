#!/usr/bin/env python
import numpy
import math
import os
import sys
import pickle
#import collections
#import types
#import traceback
#import pprint
#pp = pprint.PrettyPrinter(indent=4)

# ROS imports
import rospy
#import tf
#from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
#from geometry_msgs.msg import Pose
import sensor_msgs.point_cloud2 as pc2

# Local imports
import pcl_helper
import pclproc

#====================== GLOBALS =====================
# Clearly this module wants to be a class
g_pcl_sub = None
g_pcl_objects_pub = None
g_pcl_table_pub = None
g_callBackCount = -1

# For testing only
g_doRunRosNode = True # For invoking RunRosNode() when run from pycharm
g_doTests = False # Invokes Test_Process_msgPCL() when file is run
g_doDebugIO = True # Saves intermediate pcd files and displays clustering plots
g_testmsgPCLFilename = "./Assets/msgPCL.pypickle" # File containing a typical Ros msgPCL, used by doTests
g_doSaveTestmsgPCL = False

#--------------------------------- ProcessPCL()
def ProcessPCL(pclpcRawIn):
    pclRecs = [] # For dev/debug display. Container for point cloud records: tuple (pclObj, pclName# )

    pclRecs.append((pclpcRawIn, "pclpcRawIn"))

    pclRecsDownSampled = pclproc.PCLProc_DownSampleVoxels(pclpcRawIn)
    pclRecs += pclRecsDownSampled
    pclpcDownSampled, pclpcDownSampledName = pclRecsDownSampled[0]

    # TODO: PassThrough Filter
    pclRecsRansac = pclproc.PCLProc_Ransac(pclpcDownSampled)
    pclRecs += pclRecsRansac

    # TODO: Extract inliers and outliers
    pclpcPassZ, pclpcPassZIn, pclpcPassZOut = pclRecsRansac[0][0], pclRecsRansac[1][0], pclRecsRansac[2][0]
    pclpcTable, pclpcObjects = pclpcPassZIn, pclpcPassZOut

    # TODO: Euclidean Clustering

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately

    return pclpcObjects, pclpcTable

#--------------------------------- Process_msgPCL()
def Process_msgPCL(msgPCL):
    # Extract pcl Raw from Ros msgPCL
    pclpcRawIn = pcl_helper.ros_to_pcl(msgPCL)

    #------- PROCESS RAW PC-------------------------
    pclpcObjects, pclpcTable = ProcessPCL(pclpcRawIn)

    # Package Processed pcl into Ros msgPCL
    msgPCLObjects = pcl_helper.pcl_to_ros(pclpcObjects)
    msgPCLTable = pcl_helper.pcl_to_ros(pclpcTable)

    return msgPCLObjects, msgPCLTable

#--------------------------------- CB_msgPCL()
def CB_msgPCL(msgPCL):
    global g_callBackCount
    g_callBackCount += 1

    cbSkip = 50
    if (g_callBackCount % cbSkip != 0):
        return;

    print "\rCBCount= {:05d}".format(g_callBackCount),
    sys.stdout.flush()

    # DevDebug save msgPCL to file for debug
    if (g_doSaveTestmsgPCL & g_callBackCount == cbSkip):
        pickle.dump(msgPCL, open(g_testmsgPCLFilename, "wb"))

    msgPCLObjects, msgPCLTable = Process_msgPCL(msgPCL)

    # Publish ROS messages
    g_pcl_objects_pub.publish(msgPCLObjects)
    g_pcl_table_pub.publish(msgPCLTable)


#====================== Main() =====================
def RunRosNode():
    print("Ros node subscribe/publish init...")

    global g_pcl_sub
    global g_pcl_objects_pub
    global g_pcl_table_pub

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    g_pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, CB_msgPCL, queue_size=1)

    # Create Publishers
    g_pcl_objects_pub = rospy.Publisher("/pcl_objects", pcl_helper.PointCloud2, queue_size=1)
    g_pcl_table_pub = rospy.Publisher("/pcl_table", pcl_helper.PointCloud2, queue_size=1)

    # Initialize color_list
    pcl_helper.get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        print("Ros.clustering node statrted.")
        rospy.spin()


#====================== Invoke RunRosNode() =====================
if ((__name__ == '__main__') & g_doRunRosNode):
    RunRosNode()


###################################### TESTS ###########################
###################################### TESTS ###########################
###################################### TESTS ###########################

#--------------------------------- Test_Process_msgPCL()
def Test_Process_msgPCL():
    msgPCL = pickle.load( open(g_testmsgPCLFilename, "rb" ) )
    msgPCLObjects, msgPCLTable = Process_msgPCL(msgPCL)


# ============ Auto invoke Test_PCLProc_*
if (g_doTests):
    Test_Process_msgPCL()
