# OBSOLETE : Consolidated into catkin_ws/src/scripts/pclproc.py

import matplotlib

def Backend_Switch(whichBackEnd):
    matplotlib.use(whichBackEnd, warn=False, force=True)
    from matplotlib import pyplot as plt
    print "Switched to:", matplotlib.get_backend()

Backend_Switch('QT4Agg')

import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import DBSCAN

# ------------------------------ PCLProc_DBScan()
def PCLProc_DBScan(ptsIn):
    # Define max_distance (eps parameter in DBSCAN())
    max_distance = 1
    db = DBSCAN(eps=max_distance, min_samples=10).fit(ptsIn)

    # Extract a mask of core cluster members
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # Extract labels (-1 is used for outliers)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)

    return core_samples_mask, labels, unique_labels


###################################### TESTS ###########################

# Define a function to generate clusters
def Test_GenerateClusters(numClusters, pts_minmax=(10, 100), x_mult=(1, 4), y_mult=(1, 3), x_off=(0, 50),
                          y_off=(0, 50)):
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

    # Convert to a single dataset in OpenCV format
    testClusters = np.float32((np.concatenate(testClustersx), np.concatenate(testClustersy))).transpose()

    # Return cluster positions
    return testClusters, testClustersx, testClustersy


#------------------------------ PlotClustersDBScan()
def PlotClustersDBScan(ptsIn, ptsInx, ptsIny, core_samples_mask, labels, unique_labels):
    n_clusters = 50
    # Plot up the results!
    min_x = np.min(ptsIn[:, 0])
    max_x = np.max(ptsIn[:, 0])
    min_y = np.min(ptsIn[:, 1])
    max_y = np.max(ptsIn[:, 1])

    fig = plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.plot(ptsIn[:,0], ptsIn[:,1], 'ko')
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.title('Original Data', fontsize = 20)

    plt.subplot(122)

    # The following is just a fancy way of plotting core, edge and outliers
    # Credit to: http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = ptsIn[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=7)

        xy = ptsIn[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=3)
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.title('DBSCAN: %d clusters found' % n_clusters, fontsize = 20)
    fig.tight_layout()
    plt.subplots_adjust(left=0.03, right=0.98, top=0.9, bottom=0.05)
    plt.show()

#------------------------------ TEST
def Test_PCLProc_DBScan():
    numClusters = 7
    testClusters, testClustersx, testClustersy = Test_GenerateClusters(numClusters)

    # INVOCATION
    core_samples_mask, labels, unique_labels = PCLProc_DBScan(testClusters)
    PlotClustersDBScan(testClusters, testClustersx, testClustersy, core_samples_mask, labels, unique_labels)


Test_PCLProc_DBScan()