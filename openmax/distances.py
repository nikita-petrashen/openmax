import torch


def cosine_distance(x, y):
    """Compute distances d_i = d(x_i, y_i)"""

    x_norm = x.norm(dim=-1)
    y_norm = y.norm(dim=-1)
    cosine_sim = torch.sqrt((x * y).sum(dim=-1)) / (x_norm * y_norm)

    return 1 - cosine_sim


def euclidean_distance(x, y):
    """Compute distances d_i = d(x_i, y_i)"""

    return (x - y).norm(dim=-1)


def pairwise_cosine_distance(x, y):
    """Compute distances d_ij = d(x_i, y_j)"""

    x_norm = x.norm(dim=-1)
    y_norm = y.norm(dim=-1)
    cosine_sim = x @ y.T / torch.outer(x_norm, y_norm)

    return 1 - cosine_sim


def pairwise_euclidean_distance(x, y):
    """Compute distances d_ij = d(x_i, y_j)"""
    
    return torch.cdist(x, y)


# TODO: determine coeff for eucl dist
def compute_distance(samples, centroid, distance_type='eucl', c=1/200):
    """
        Compute distances between class samples' embeddings and their class' centroid.

        Args:
             samples [N x dim], torch.tensor(float): embeddings of the correctly classified train dataset samples,
             centroind [dim, ], torch.tensor(float): mean vector of each class from :samples:
        Returns:
             distances [N, ] (torch.tensor(float))
    """

    if distance_type == 'eucos':
        distances = c * euclidean_distance(samples, centroid) + cosine_distance(samples, centroid)
    elif distance_type == 'eucl':
        distances = euclidean_distance(samples, centroid)
    elif distance_type == 'cos':
        distances = cosine_distance(samples, centroid)
    else:
        raise ValueError(f"Expected argument 'distance_type' to be one of: "
                         f"eucl, cos, eucos, got {distance_type} instead")

    return distances


# TODO: determine coeff for eucl dist
def compute_pairwise_distance(samples, centroids, distance_type='eucl', c=1/200):
    """
        Compute distances between a bunch of samples' embeddings and every class' centroid.

        Args:
             samples [N_s x dim], torch.tensor(float): embeddings of the correctly classified train dataset samples,
             centroind [N_c x dim], torch.tensor(float): mean vector of each class from :samples:
        Returns:
             distances [N_s x N_c], torch.tensor(float)
    """

    if distance_type == 'eucos':
        distances = c * pairwise_euclidean_distance(samples, centroids) + pairwise_cosine_distance(samples, centroids)
    elif distance_type == 'eucl':
        distances = pairwise_euclidean_distance(samples, centroids)
    elif distance_type == 'cos':
        distances = pairwise_cosine_distance(samples, centroids)
    else:
        raise ValueError(f"Expected argument 'distance_type' to be one of: "
                         f"eucl, cos, eucos, got {distance_type} instead")

    return distances
