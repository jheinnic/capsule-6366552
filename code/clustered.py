from sklearn.metrics import pairwise
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from skimage.io import imread, imsave
import numpy as np
import numpy.random as random

def sample_unique_pixel_avg_distance(source_file, pixel_sample_size, metric="cityblock"):
    rng = random.default_rng()
    # rr = imread('../../StorageResearch/' + 'ee19f416b1735c6ec5fc2ff3c1524a761032d001/1050_accum16_20200212_163258.png')
    rr = imread(source_file)
    rf = rr.flatten()
    ru = np.unique(rf)[1:]
    rs = [(ii, np.argwhere(rr == ii)) for ii in ru]
    rx = [[ii[0], rng.permutation(ii[1])[:pixel_sample_size]] if len(ii[1]) > pixel_sample_size else ii for ii in rs]
    n_clusters = len(rx)
    avg_dist = np.zeros((n_clusters, n_clusters))
    for ii in range(n_clusters):
        for jj in range(n_clusters):
            avg_dist[ii, jj] = pairwise_distances(rx[ii][1], rx[jj][1],metric=metric).mean()
    avg_dist /= avg_dist.max()
    pixels_by_value = [(ii, np.nonzero(rr == ii)) for ii in ru]
    return avg_dist, pixels_by_value

    
def cluster_by_distance_matrix(avg_dist, pixel_value_count, cluster_count=2, linkage="complete"):
    cluster = AgglomerativeClustering(n_clusters=cluster_count, affinity="precomputed", linkage=linkage)  
    results = cluster.fit_predict(avg_dist)  
    labels = np.arange(0, pixel_value_count)
    cluster_labels = np.column_stack((labels, results))
    return cluster_labels


def split_cluster_submatrices(
    source_file, pixels_by_value, cluster_labels, group_name, cluster_count=2, output_template='splits/rr_{group_name}-{ii}.png'
):
    rr = imread(source_file)
    cluster_pixel_counts = [0] * cluster_count
    cluster_submatrices = [np.zeros(rr.shape, dtype=np.uint16) for i in cluster_labels]
    for ii in range(0, len(cluster_labels):
        (inst_idx, class_id) = cluster_labels[ii]
        (pixels_value, pixel_xys) = pixels_by_value[inst_idx]
        cluster_submatrices[class_id][pixel_xys] = pixels_value
        cluster_pixel_counts[class_id] = cluster_pixel_counts[class_id] + 1
    validation_merge = 0
    for ii in range(0, cluster_count):
        imsave(f'splits/rr_{id}-{ii}.png', cluster_submatrices[ii])
        validation_merge = validation_merge + cluster_submatrices[ii]
    print((validation_merge == rr).all())
    return cluster_submatrices, cluster_pixel_counts


def split_to_cliques(source_file, group_name, sample_size, cluster_count=2, metric="cityblock", linkage="complete"):
    (avg_dist, pixels_by_value) = sample_unique_pixel_avg_distance(source_file, sample_size, metric=metric)
    cluster_labels = cluster_by_distance_matrix(avg_dist, len(pixels_by_value), cluster_count=cluster_count, linkage=linkage)
    return split_cluster_submatrices(source_file, pixels_by_value, cluster_labels, group_name, cluster_count=cluster_count)