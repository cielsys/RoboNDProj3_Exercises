import pickle
#print("CWD:", os.getcwd())
#absPath = os.path.abspath('../src/sensor_stick')
#print("absPath", absPath)
#sys.path.insert(0, absPath)
#sys.path.insert(0, "/home/cl/AAAProjects/AAAUdacity/roboND/Proj3_3dPerception/RoboND-Perception-Exercises/catkin_ws3/src/sensor_stick/src/sensor_stick/")
#print("sys.path",sys.path)
#from sensor_stick import pcl_helper
#import sensor_stick
import sensor_stick.pcl_helper as pcl_helper

#--------------------------------- pickle templates
def Camera_SaveCalFile(calFileName, dictCameraCalVals):
    pickle.dump(dictCameraCalVals, open(calFileName, "wb" ) )

def LoadFile_msgPCL(fileNameIn = "msgPCL.pypickle"):
    retVal = pickle.load( open(fileNameIn, "rb" ) )
    return(retVal)

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

    # Create a VoxelGrid filter object for our input point cloud
    vox = pclpcIn.make_voxel_grid_filter()
    voxelSize = 0.01
    vox.set_leaf_size(voxelSize, voxelSize, voxelSize)

    # TODO: Voxel Grid Downsampling
    # Call the filter function to obtain the resultant downsampled point cloud
    pclpcVoxels = vox.filter()
    pclRecs.append((pclpcVoxels, "pclpcDownSampled"))
