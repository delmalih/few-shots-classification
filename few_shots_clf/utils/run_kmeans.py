##########################
# Imports
##########################


# Global
import numpy as np


##########################
# Function
##########################


def run_kmeans(points, k, eps=0.5, verbose=False):
    """[summary]

    Args:
        points ([type]): [description]
        k ([type]): [description]

    Returns:
        [type]: [description]
    """
    new_centroids = _initialize_centroids(points, k)
    old_centroids = np.zeros_like(new_centroids)
    shift = np.linalg.norm(new_centroids - old_centroids)
    while shift > eps:
        old_centroids = new_centroids.copy()
        closest = _closest_centroid(points, old_centroids)
        new_centroids = _move_centroids(points, closest, old_centroids)
        shift = np.linalg.norm(new_centroids - old_centroids)
        if verbose:
            print(f"KMeans step: {shift}")
    return closest, new_centroids


def _initialize_centroids(points, k):
    centroids_idx = np.random.choice(range(len(points)),
                                     size=k,
                                     replace=False)
    centroids = points[centroids_idx]
    return centroids


def _closest_centroid(points, centroids):
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


def _move_centroids(points, closest, centroids):
    return np.array([points[closest == k].mean(axis=0)
                     for k in range(centroids.shape[0])])
