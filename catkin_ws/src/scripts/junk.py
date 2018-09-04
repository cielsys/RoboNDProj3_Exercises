import pickle

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
