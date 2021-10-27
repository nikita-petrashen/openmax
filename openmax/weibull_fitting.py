import numpy as np
import scipy.stats
from .distances import compute_distance


class WeibullFitting:
    """
        This class holds Weibull models for each category in the dataset.

        Params:
            tailsize (int): number of farthest samples from the MAV (see paper) to fit the Weibull distribution on

        based on https://github.com/ashafaei/OD-test/blob/8252aace84e2ae1ab95067876985f62a1060aad6/methods/openmax.py#L37
    """
    # some constants from the original repo
    translation = 10000
    a = 1
    loc = 0

    def __init__(self, tailsize=10, num_classes=31):
        self.tailsize = tailsize
        self.num_classes = num_classes
        # some constant from the original repo
        self.translation = 10000
        self.weibull_models = None

    def fit_high(self, values):
        """ Fit Weibull distribution's parameters """
        tail = np.sort(values)[-self.tailsize:]
        min_val = tail[0]
        tail = [v + self.translation - min_val for v in tail]
        params = scipy.stats.exponweib.fit(tail, floc=0, f0=1)
        weibull_model = {"c": params[1], "scale": params[3], "min_val": min_val}

        return weibull_model

    def fit_all_models(self, distances):
        """
        Fit Weibull models for each of the classes.
            Args:
                distances (list of torch.tensors):
                    [tensor[N_cls1, ], tensor[N_cls2, ], ...] list of [distances to MAV] for each class
        """
        weibull_models = []
        for i in range(self.num_classes):
            weibull_models.append(self.fit_high(distances[i]))

        self.weibull_models = weibull_models

        return weibull_models

    @classmethod
    def w_score(cls, distances, weibull_model):
        """
        For a list of distances, compute the probabilities according to the Weibull model of a given class.
            Args:
                weibull_model: dictionary returned by :meth: fit_high
        """
        c, scale, min_val = weibull_model.values()
        values = distances + cls.translation - min_val
        w_score = scipy.stats.exponweib.cdf(values, a=cls.a, c=c, loc=cls.loc, scale=scale)

        return w_score

    def compute_centroids_and_distances(self, samples, distance_type='eucl'):
        """
        Compute centroids and distances from samples to centroids.
            Args:
                samples (list of torch.tensors):
                    [tensor[N_cls1, dim], tensor[N_cls2, dim], ...] list of [correctly classified samples] for each class.
        """
        distances, centroids = [], []
        for i in range(self.num_classes):
            class_samples = samples[i]
            class_centroid = class_samples.mean(dim=0)
            class_distances = compute_distance(class_samples, class_centroid, distance_type=distance_type)
            distances.append(class_distances)
            centroids.append(class_centroid)

        return distances, centroids



