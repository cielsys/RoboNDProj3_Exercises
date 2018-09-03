import pickle

#--------------------------------- pickle templates
def Camera_SaveCalFile(calFileName, dictCameraCalVals):
    pickle.dump(dictCameraCalVals, open(calFileName, "wb" ) )

def LoadFile_msgPCL(fileNameIn = "msgPCL.pypickle"):
    retVal = pickle.load( open(fileNameIn, "rb" ) )
    return(retVal)
