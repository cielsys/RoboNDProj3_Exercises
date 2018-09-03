import datetime
import subprocess
import pcl

def PCLProc_Ransac(pclpcIn):
    pclRecs = [] # For dev/debug display. Container for point cloud records: tuple (pclObj, pclName)
    pclRecs.append((pclpcIn, "pclpcIn"))

    # Create a VoxelGrid filter object for our input point cloud
    vox = pclpcIn.make_voxel_grid_filter()
    voxelSize = 0.01
    vox.set_leaf_size(voxelSize, voxelSize, voxelSize)

    # Call the filter function to obtain the resultant downsampled point cloud
    pclpcVoxels = vox.filter()
    pclRecs.append((pclpcVoxels, "pclpcDownSampled"))

    # Create a PassThrough filter object.
    filPassthrough = pclpcVoxels.make_passthrough_filter()

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

    # Not really needed here because we have no noise
    # Much like the previous filters, we start by creating a filter object:
    #outlier_filter = cloud_filtered.make_statistical_outlier_filter(1)

    # Set the number of neighboring points to analyze for any given point
    #outlier_filter.set_mean_k(50)

    # Set threshold scale factor
    #x = 1.0

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    #outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    #cloud_filtered = outlier_filter.filter()
    return(pclRecs)

def PCLProc_Noise(pclpcIn):
    pclRecs = [] # For dev/debug display. Container for point cloud records: tuple (pclObj, pclName)

    fil = pclpcIn.make_statistical_outlier_filter()
    k_numNeighborsToCheck = 50
    k_threshScaleFactor = 1.0
    fil.set_mean_k(k_numNeighborsToCheck)
    fil.set_std_dev_mul_thresh(k_threshScaleFactor)

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
    # Load Point Cloud file
    dirNameIn = "./Assets/pcdIn/"
    fileNameBaseIn = 'tabletop.pcd'
    fileNameIn = dirNameIn + fileNameBaseIn
    pclpcIn = pcl.load_XYZRGB(fileNameIn)

    pclRecs = PCLProc_Ransac(pclpcIn)
    dirNameOut = "./Assets/pcdOut/"
    SavePCLs(pclRecs, dirNameOut, useTimeStamp=True)

#============ Auto invoke Test_PCLProc_Ransac()
#Test_PCLProc_Ransac()


# ----------------------- Test_PCLProc_Noise()
def Test_PCLProc_Noise():
    # Load Point Cloud file
    dirNameIn = "./Assets/pcdIn/"
    fileNameBaseIn = 'table_scene_lms400.pcd'
    fileNameIn = dirNameIn + fileNameBaseIn

    pclpcIn = pcl.load(fileNameIn)

    pclRecs = PCLProc_Noise(pclpcIn)
    dirNameOut = "./Assets/pcdOut/"
    SavePCLs(pclRecs, dirNameOut, useTimeStamp=True)


# ============ Auto invoke Test_PCLProc_Noise()
Test_PCLProc_Noise()