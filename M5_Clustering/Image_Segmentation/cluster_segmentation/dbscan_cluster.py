import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN, estimate_bandwidth
from sklearn.preprocessing import StandardScaler


def dbscan_cluster(img):
    # Flatten Image
    nr_color_cols = 3
    if len(img.shape) < 3:
        nr_color_cols = 1

    flat_image = img.reshape((-1, nr_color_cols))

    # Normalize Features
    scaler = StandardScaler()
    flat_image = scaler.fit_transform(flat_image)

    # meanshift
    eps = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
    eps = 3
    model = DBSCAN(eps=eps, min_samples=5, n_jobs=1)
    model.fit(flat_image)
    labeled = model.labels_

    # get number of segments
    labels = np.unique(labeled)
    segments = labels
    print('Number of segments: ', segments.shape[0])

    # Assign the aggregated color to each segment

    flat_image = scaler.inverse_transform(flat_image)
    flat_segm_image = np.zeros(flat_image.shape)
    for i, label in enumerate(labels):
        row_sel = labeled == label
        cluster_color = np.mean(flat_image[row_sel, :], axis=0)
        flat_segm_image[row_sel, :] = cluster_color

    flat_segm_image = np.uint8(flat_segm_image)
    segmented_image = flat_segm_image.reshape((img.shape))

    return segmented_image, flat_segm_image, labeled
