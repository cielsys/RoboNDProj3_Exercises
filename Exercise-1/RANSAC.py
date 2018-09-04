# OBSOLETE : Consolidated into catkin_ws/src/scripts/pclproc.py

import datetime
import subprocess
import pcl

#----------------------- PCLProc_DownSampleVoxels()
def PCLProc_DownSampleVoxels(pclpcIn):
    # Create a VoxelGrid filter object for our input point cloud
    vox = pclpcIn.make_voxel_grid_filter()
    voxelSize = 0.01
    vox.set_leaf_size(voxelSize, voxelSize, voxelSize)
    # Call the filter function to obtain the resultant downsampled point cloud
    pclpcDownSampled = vox.filter()
    pclRecs = [(pclpcDownSampled, "pclpcDownSampled")]
    return pclRecs

#----------------------- PCLProc_Ransac()
def PCLProc_Ransac(pclpcIn):
    pclRecs = [] # For dev/debug display. Container for point cloud records: tuple (pclObj, pclName)
    #pclRecs.append((pclpcIn, "pclpcIn"))

    #pclpcDownSampled = PCLProc_DownSampleVoxels(pclpcIn)
    #pclRecs.append((pclpcDownSampled, "pclpcDownSampled"))

    # Create a PassThrough filter object.
    filPassthrough = pclpcIn.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    axis_min = 0.6
    axis_max = 1.1
    filPassthrough.set_filter_field_name(filter_axis)
    filPassthrough.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    pclpcPassZ = filPassthrough.filter()
    pclRecs.append((pclpcPassZ, "pclpcPassZ"))

    # RANSAC plane segmentation
    # Create the segmentation object
    seg = pclpcPassZ.make_segmenter()

    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance
    # for segmenting the table
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # Extract inliers
    pclpcPassZIn = pclpcPassZ.extract(inliers, negative=False)
    pclpcPassZOut = pclpcPassZ.extract(inliers, negative=True)

    pclRecs.append((pclpcPassZIn, "pclpcPassZIn"))
    pclRecs.append((pclpcPassZOut, "pclpcPassZOut"))

    return(pclRecs)

#----------------------- PCLProc_Noise()
def PCLProc_Noise(pclpIn):
    pclRecs = [] # For dev/debug display. Container for point cloud records: tuple (pclObj, pclName)

    fil = pclpIn.make_statistical_outlier_filter()
    numNeighborsToCheck = 50
    threshScaleFactor = 1.0
    fil.set_mean_k(numNeighborsToCheck)
    fil.set_std_dev_mul_thresh(threshScaleFactor)

    pclpNoiseInliers = fil.filter()
    fil.set_negative(True)
    pclpNoiseOutliers = fil.filter()

    pclRecs.append((pclpNoiseInliers, "pclpNoiseInliers"))
    pclRecs.append((pclpNoiseOutliers, "pclpNoiseOutliers"))
    return(pclRecs)


###################################### TESTS ###########################

#----------------------- SavePCLs()
def SavePCLs(pclRecs, dirNameOut, useTimeStamp=True):
    if (useTimeStamp):
        strDT = "_{:%Y-%m-%dT%H:%M:%S}".format(datetime.datetime.now())
    else:
        strDT = ""

    for pclObj, pclName in pclRecs:
        extOut = ".pcd"
        fileNameOutBase = pclName +  strDT + extOut
        fileNameOut= dirNameOut + fileNameOutBase
        pcl.save(pclObj, fileNameOut)
        #subprocess.call(["pcl_viewer", fileNameOut])

#----------------------- Test_PCLProc_Ransac()
def Test_PCLProc_Ransac():
    pclRecs = [] # For dev/debug display. Container for point cloud records: tuple (pclObj, pclName)
    # Load Point Cloud file
    dirNameIn = "./Assets/pcdIn/"
    fileNameBaseIn = 'tabletop.pcd'
    fileNameIn = dirNameIn + fileNameBaseIn
    pclpcInRaw = pcl.load_XYZRGB(fileNameIn)
    pclRecs.append((pclpcInRaw, "pclpcInRaw"))

    pclRecsDownSampled = PCLProc_DownSampleVoxels(pclpcInRaw)
    pclpcDownSampled, pclpcDownSampledName = pclRecsDownSampled[0]
    pclRecs += pclRecsDownSampled

    pclRecsRansac = PCLProc_Ransac(pclpcDownSampled)
    pclRecs += pclRecsRansac

    dirNameOut = "./Assets/pcdOut/"
    SavePCLs(pclRecs, dirNameOut, useTimeStamp=True)

# ----------------------- Test_PCLProc_Noise()
def Test_PCLProc_Noise():
    pclRecs = [] # For dev/debug display. Container for point cloud records: tuple (pclObj, pclName)

    # Load Point Cloud file
    dirNameIn = "./Assets/pcdIn/"
    fileNameBaseIn = 'table_scene_lms400.pcd'
    fileNameIn = dirNameIn + fileNameBaseIn

    pclpRaw = pcl.load(fileNameIn)
    pclRecs.append((pclpRaw, "pclpcInRaw"))

    pclRecsDownSampled = PCLProc_DownSampleVoxels(pclpRaw)
    pclpDownSampled, pclpDownSampledName = pclRecsDownSampled[0]
    pclRecs += pclRecsDownSampled

    pclRecsNoise = PCLProc_Noise(pclpDownSampled)
    pclRecs += pclRecsNoise

    dirNameOut = "./Assets/pcdOut/"
    SavePCLs(pclRecs, dirNameOut, useTimeStamp=True)

# ============ Auto invoke Test_PCLProc_Noise()
#Test_PCLProc_Ransac()
Test_PCLProc_Noise()