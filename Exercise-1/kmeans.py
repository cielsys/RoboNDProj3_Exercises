import matplotlib

def Backend_Switch(whichBackEnd):
    matplotlib.use(whichBackEnd, warn=False, force=True)
    from matplotlib import pyplot as plt
    print "Switched to:", matplotlib.get_backend()

Backend_Switch('QT4Agg')
#quit()
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Define a function to generate clusters
def Test_GenerateClusters(numClusters, pts_minmax=(10, 100), x_mult=(1, 4), y_mult=(1, 3), x_off=(0, 50), y_off=(0, 50)):
    """
    # n_clusters = number of clusters to generate
    # pts_minmax = range of number of points per cluster
    # x_mult = range of multiplier to modify the size of cluster in the x-direction
    # y_mult = range of multiplier to modify the size of cluster in the y-direction
    # x_off = range of cluster position offset in the x-direction
    # y_off = range of cluster position offset in the y-direction
    """

    # Initialize some empty lists to receive cluster member positions
    testClustersx = []
    testClustersy = []
    # Genereate random values given parameter ranges
    n_points = np.random.randint(pts_minmax[0], pts_minmax[1], numClusters)
    x_multipliers = np.random.randint(x_mult[0], x_mult[1], numClusters)
    y_multipliers = np.random.randint(y_mult[0], y_mult[1], numClusters)
    x_offsets = np.random.randint(x_off[0], x_off[1], numClusters)
    y_offsets = np.random.randint(y_off[0], y_off[1], numClusters)

    # Generate random clusters given parameter values
    for idx, npts in enumerate(n_points):
        xpts = np.random.randn(npts) * x_multipliers[idx] + x_offsets[idx]
        ypts = np.random.randn(npts) * y_multipliers[idx] + y_offsets[idx]
        testClustersx.append(xpts)
        testClustersy.append(ypts)

    # Return cluster positions
    return testClustersx, testClustersy


def PlotClusters(ptsIn, ptsInx, ptsIny, kmeansClustersx, kmeansClustersy):
    # Plot up a comparison of original clusters vs. k-means clusters
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(121)
    min_x = np.min(ptsIn[:, 0])
    max_x = np.max(ptsIn[:, 0])
    min_y = np.min(ptsIn[:, 1])
    max_y = np.max(ptsIn[:, 1])
    for idx, xpts in enumerate(ptsInx):
        plt.plot(xpts, ptsIny[idx], 'o')
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.title('Original Clusters', fontsize=20)
    plt.subplot(122)

    for idx, xpts in enumerate(kmeansClustersx):
        plt.plot(xpts, kmeansClustersy[idx], 'o')
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.title('k-means Clusters', fontsize=20)
    fig.tight_layout()
    plt.subplots_adjust(left=0.03, right=0.98, top=0.9, bottom=0.05)
    plt.show()

#------------------------------ PCLProc_KMeans()
def PCLProc_KMeans(ptsIn):
    # Define k-means parameters
    # Number of clusters to define
    numClusters = 7
    # Maximum number of iterations to perform
    max_iter = 10
    # Accuracy criterion for stopping iterations
    epsilon = 1.0
    # Define criteria in OpenCV format
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    # Call k-means algorithm on your dataset
    compactness, label, center = cv2.kmeans(ptsIn, numClusters, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # NERVOUS - The provided call to kmeans had 'None' param not wanted in this version
    # compactness, label, center = cv2.kmeans(data, k_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Define some empty lists to receive k-means cluster points
    kmeansClustersx = []
    kmeansClustersy = []

    # Extract k-means clusters from output
    for idx in range(numClusters):
        kmeansClustersx.append(ptsIn[label.ravel() == idx][:, 0])
        kmeansClustersy.append(ptsIn[label.ravel() == idx][:, 1])

    return kmeansClustersx, kmeansClustersy


def Test_PCLProc_KMeans():
    numClusters = 7
    ptsInx, ptsIny = Test_GenerateClusters(numClusters)
    # Convert to a single dataset in OpenCV format
    ptsIn = np.float32((np.concatenate(ptsInx), np.concatenate(ptsIny))).transpose()

    # INVOCATION
    kmeansClustersx, kmeansClustersy = PCLProc_KMeans(ptsIn)
    PlotClusters(ptsIn, ptsInx, ptsIny, kmeansClustersx, kmeansClustersy)


Test_PCLProc_KMeans()