import torch
import numpy as np

from .distances import compute_pairwise_distance
from .weibull_fitting import WeibullFitting


class OpenMax:
    """
        Given Weibull models for each class, compute OpenMax logits $\hat v$.

        Params:
            centroids (torch.tensor[num_classes, dim]): centroids for each class.
            weibull_models (list of dicts): list of Weibull models computed by WeibullFitting class.
            alpha (int): number of classes for scores recalibration (see paper).
            distance_type (str): distance type to use (should be the same as used in WeibullFitting).
    """
    def __init__(self, centroids, weibull_models, alpha=10, distance_type='eucl'):
        if distance_type not in ['eucl', 'cos', 'eucos']:
            raise ValueError(f"Expected argument 'distance_type' to be one of: "
                             f"eucl, cos, eucos, got {distance_type} instead")
        self.centroids = torch.stack(centroids)
        self.weibull_models = weibull_models
        self.alpha = alpha
        self.distance_type=distance_type
        self.num_classes = self.centroids.shape[0]

    def recalibrate_logits(self, logits, embs=None):
        """
        Compute recalibrated logits.
            Args:
                logits (torch.tensor[N_samples, num_classes]): tensor of inputs to the SoftMax layer of the network.
                embs (torch.tensor[N_samples, dim]): if WeibullFitting was done on some other intermediate
                                embeddings in the network, use them to compute recalibration coeffs instead of logits.
        """
        if embs is None:
            embs = logits
        logits = logits.detach().cpu().squeeze()
        embs = embs.detach().cpu().squeeze()
        top_preds = logits.argsort(-1, descending=True)[:, :self.alpha][:, None]
        row_idx = np.arange(logits.shape[0])[:, None]
        alpha_coeffs = np.zeros_like(logits)
        for i in range(self.alpha):
            alpha_coeffs[row_idx, top_preds[..., i]] = (self.alpha - i) / self.alpha
        
        distances = compute_pairwise_distance(embs, self.centroids, distance_type=self.distance_type)
        weibull_probs = np.zeros_like(distances)
        for i in range(self.num_classes):
            weibull_probs[:, i] = WeibullFitting.w_score(distances[:, i], self.weibull_models[i])

        logits_hat = logits * (1 - alpha_coeffs * weibull_probs)
        logit_unk = (logits  - logits_hat).sum(dim=-1).unsqueeze(-1)
        logits_hat = torch.cat([logit_unk, logits_hat], dim=-1)

        return logits_hat

    @staticmethod
    def compute_probs(logits_hat):
        """ Compute recalibrated probabilities"""
        return torch.nn.functional.softmax(logits_hat, dim=-1)
