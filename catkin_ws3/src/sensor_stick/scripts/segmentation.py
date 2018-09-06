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
import pcl

# Local imports
import pcl_helper

#import sensor_stick.pcl_helper
import pclproc

#====================== GLOBALS =====================
# Clearly this module wants to be a class
g_pcl_sub = None
g_pcl_objects_pub = None
g_pcl_table_pub = None
g_pcl_cluster_pub = None
g_callBackCount = -1
g_callBackSkip = 10 # How many callbacks to skip until actual processing

# For testing only
g_doRunRosNode = False # For invoking RunRosNode() when run from pycharm
g_doTests = True # Invokes Test_Process_msgPCL() when file is run
g_testmsgPCLFilename = "./Assets/msgPCL" # + "num..pypickle" # File containing a typical Ros msgPCL, used by doTests
g_testrawPCLFilename = "./Assets/rawPCL" # + "num.pypickle" # File containing a typical rawPCL as unpacked my pcl_helper used by doTests
g_dumpCountTestmsgPCL = 0 # How many debug msgPCL files to dump. Normally 0
g_dumpCountTestrawPCL = 0 # How many debug rawPCL files to dump. Normally 0

#--------------------------------- ProcessPCL()
def Process_rawPCL(pclpcRawIn):
    pclRecs = [] # For dev/debug display. Container for point cloud records: tuple (pclObj, pclName# )

    # Initialize color_list
    pcl_helper.get_color_list.color_list = []

    pclRecs.append((pclpcRawIn, "pclpcRawIn"))

    pclRecsDownSampled = pclproc.PCLProc_DownSampleVoxels(pclpcRawIn)
    pclRecs += pclRecsDownSampled
    pclpcDownSampled, pclpcDownSampledName = pclRecsDownSampled[0]

    # PassThrough Filter
    pclRecsRansac = pclproc.PCLProc_Ransac(pclpcDownSampled)
    pclRecs += pclRecsRansac

    # Extract inliers and outliers
    pclpcPassZ, pclpcPassZIn, pclpcPassZOut = pclRecsRansac[0][0], pclRecsRansac[1][0], pclRecsRansac[2][0]
    pclpcTable, pclpcObjects = pclpcPassZIn, pclpcPassZOut

    # Euclidean Clustering
    pclpObjectsNoColor = pcl_helper.XYZRGB_to_XYZ(pclpcObjects)
    pclpcClusters = pclproc.PCLProc_ExtractClusters(pclpObjectsNoColor)

    return pclpcObjects, pclpcTable, pclpcClusters


#--------------------------------- Process_msgPCL()
def Process_msgPCL(msgPCL):
    global g_dumpCountTestmsgPCL
    global g_dumpCountTestrawPCL

    # DevDebug save msgPCL to file for debug
    if (g_dumpCountTestmsgPCL > 0):
        g_dumpCountTestmsgPCL -= 1
        fileNameOut = g_testmsgPCLFilename + str(g_dumpCountTestmsgPCL)  + ".pypickle"
        pickle.dump(msgPCL, open(fileNameOut, "wb"))

    # Extract pcl Raw from Ros msgPCL
    pclpcRawIn = pcl_helper.ros_to_pcl(msgPCL)

    # DevDebug save msgPCL to file for debug
    if (g_dumpCountTestrawPCL > 0):
        g_dumpCountTestrawPCL -= 1
        fileNameOut = g_testrawPCLFilename + str(g_dumpCountTestrawPCL)  + ".pypickle"
        pickle.dump(pclpcRawIn, open(fileNameOut, "wb"))

    #------- PROCESS RAW PCL-------------------------
    pclpcObjects, pclpcTable, pclpcClusters = Process_rawPCL(pclpcRawIn)

    # Package Processed pcls into Ros msgPCL
    msgPCLObjects = pcl_helper.pcl_to_ros(pclpcObjects)
    msgPCLTable = pcl_helper.pcl_to_ros(pclpcTable)
    msgPCLClusters = pcl_helper.pcl_to_ros(pclpcClusters)

    return msgPCLObjects, msgPCLTable, msgPCLClusters

#--------------------------------- CB_msgPCL()
def CB_msgPCL(msgPCL):
    global g_callBackCount
    g_callBackCount += 1

    if (g_callBackCount % g_callBackSkip != 0):
        return;

    print "\rCBCount= {:05d}".format(g_callBackCount),
    sys.stdout.flush()

    msgPCLObjects, msgPCLTable, msgPCLClusters = Process_msgPCL(msgPCL)

    # Publish ROS messages
    g_pcl_objects_pub.publish(msgPCLObjects)
    g_pcl_table_pub.publish(msgPCLTable)
    g_pcl_cluster_pub.publish(msgPCLClusters)


#====================== Main() =====================
def RunRosNode():
    '''
    ROS segmentation node initialization
    '''
    print("OS segmentation node initializatiing...")

    global g_pcl_sub
    global g_pcl_objects_pub
    global g_pcl_table_pub
    global g_pcl_cluster_pub

    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    g_pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, CB_msgPCL, queue_size=1)

    # Create Publishers
    g_pcl_objects_pub = rospy.Publisher("/pcl_objects", pcl_helper.PointCloud2, queue_size=1)
    g_pcl_table_pub = rospy.Publisher("/pcl_table", pcl_helper.PointCloud2, queue_size=1)
    g_pcl_cluster_pub = rospy.Publisher("/pcl_cluster", pcl_helper.PointCloud2, queue_size=1)

    # Initialize color_list
    #pcl_helper.get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        print("ROS segmentation node running.")
        rospy.spin()


#====================== Invoke RunRosNode() =====================
if ((__name__ == '__main__') & g_doRunRosNode):
    RunRosNode()


###################################### TESTS ###########################
###################################### TESTS ###########################
###################################### TESTS ###########################

#--------------------------------- Test_Process_msgPCL()
def Test_Process_msgPCL():
    dumpIndex = 0
    fileNameIn = g_testmsgPCLFilename + str(dumpIndex) + ".pypickle"
    msgPCL = pickle.load( open(fileNameIn, "rb" ) )
    msgPCLObjects, msgPCLTable, pclpcClusters = Process_msgPCL(msgPCL)

#--------------------------------- Test_Process_rawPCL()
def Test_Process_rawPCL():
    dumpIndex = 0
    fileNameIn = g_testrawPCLFilename + str(dumpIndex) + ".pypickle"
    pclpcRawIn = pickle.load( open(fileNameIn, "rb" ) )
    pclpcObjects, pclpcTable, pclpcClusters = Process_rawPCL(pclpcRawIn)


# ============ Auto invoke Test_PCLProc_*
if (g_doTests):
    Test_Process_rawPCL()
    #Test_Process_msgPCL() # Depends on -> pcl_helper.py -> Active ROS environment
