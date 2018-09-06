#!/usr/bin/env python
import numpy as np
import pickle
import sys
import os
import rospy

# Ros package imports
from sensor_stick.pcl_helper import *
from sensor_stick.training_helper import spawn_model
from sensor_stick.training_helper import delete_model
from sensor_stick.training_helper import initial_setup
from sensor_stick.training_helper import capture_sample
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from sensor_stick.srv import GetNormals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2

# Local imports
import pclproc

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

def CaptureFeaturesxxx():
    rospy.init_node('capture_node')

    models = [\
       'beer',
       'bowl',
       'create',
       'disk_part',
       'hammer',
       'plastic_cup',
       'soda_can']

    # Disable gravity and delete the ground plane
    initial_setup()
    labeled_features = []

    numCapturesPerModel = 4
    numRetriesPerCapture = 3
    for model_name in models:
        print("Feature capture on model '{}' ({}) target numAttempts".format(model_name, numCapturesPerModel))
        spawn_model(model_name)

        for i in range(numCapturesPerModel):
            # make five attempts to get a valid a point cloud then give up
            sample_was_good = False
            try_count = 0

            while not sample_was_good and try_count < numRetriesPerCapture:
                sample_cloud = capture_sample()
                sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    print('Invalid cloud detected')
                    try_count += 1
                else:
                    sample_was_good = True

            if (sample_was_good):
                # Extract histogram features
                chists = pclproc.compute_color_histograms(sample_cloud, doConvertToHSV=True)

                normals = get_normals(sample_cloud)
                nhists = pclproc.compute_normal_histograms(normals)

                feature = np.concatenate((chists, nhists))
                labeled_features.append([feature, model_name])

        delete_model()

    pickle.dump(labeled_features, open('training_set.sav', 'wb'))

#--------------------------------- Test_ImgRec_CalHistogram()
def CaptureCloud(numRetriesPerCapture = 3):
    try_count = 0
    sample_was_good = False

    while not sample_was_good and try_count < numRetriesPerCapture:
        print(str(try_count)),
        sys.stdout.flush()

        sample_cloud = capture_sample()
        sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

        # Check for invalid clouds.
        if sample_cloud_arr.shape[0] == 0:
            print('*BAD,'),
            try_count += 1
        else:
            print('good,'),
            sample_was_good = True
        sys.stdout.flush()

    return sample_was_good, sample_cloud

#--------------------------------- CaptureFeaturesOfModel()
def CaptureFeaturesOfModel(modelName, numCapturesReq, numRetriesPerCapture):
    labeledFeaturesModel = []
    spawn_model(modelName)

    print('Capclouds({}):'.format(numCapturesReq)),
    for i in range(numCapturesReq):
        sample_was_good, sample_cloud = CaptureCloud(numRetriesPerCapture)
        if (sample_was_good):
            # Extract histogram features
            chists = pclproc.compute_color_histograms(sample_cloud, doConvertToHSV=True)
            normals = get_normals(sample_cloud)
            nhists = pclproc.compute_normal_histograms(normals)

            feature = np.concatenate((chists, nhists))
            labeledFeaturesModel.append([feature, modelName])

    delete_model()
    return labeledFeaturesModel


#--------------------------------- Test_ImgRec_CalHistogram()
def CaptureFeaturesOfModelList(modelNames):
    labeledFeaturesModelAccum = []

    numRetriesPerCapture = 3
    for modelName, numCaps in modelNames:
        print ("CaptureFeatures({:11}):".format(modelName) ),
        sys.stdout.flush()
        labeledFeaturesModel = CaptureFeaturesOfModel(modelName, numCaps, numRetriesPerCapture)
        print " numCaptured = ", str(len(labeledFeaturesModel))
        labeledFeaturesModelAccum += labeledFeaturesModel

    fileNameOut = 'training_set.sav'
    print("Saving training set file {}".format(os.path.abspath(fileNameOut)))
    pickle.dump(labeledFeaturesModelAccum, open(fileNameOut, 'wb'))

#--------------------------------- Main()
def Main():
    rospy.init_node('capture_node')
    initial_setup()# Disable gravity and delete the ground plane
    print("capture_features.py Main() started.")
    modelNames = [('beer', 10), ('bowl', 10), ('create', 10), ('disk_part', 10), ('hammer', 10), ('plastic_cup', 10), ('soda_can', 10)]
    CaptureFeaturesOfModelList(modelNames)

if __name__ == '__main__':
    Main()
