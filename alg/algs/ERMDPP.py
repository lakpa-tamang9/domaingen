# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import random
from scipy.spatial.distance import pdist, squareform


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ERMDPP(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, args):
        super(ERMDPP, self).__init__(args)
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes,
            self.featurizer.in_features,
            args.classifier,
        )
        self.epochs = args.max_epoch
        self.network = nn.Sequential(self.featurizer, self.classifier)

    def custom_kernel(self, X):
        """
        Compute a custom kernel matrix using pairwise Euclidean distances.
        """
        pairwise_dists = squareform(pdist(X, "euclidean"))
        # Convert distances to a similarity measure (e.g., Gaussian similarity)
        gamma = 0.8
        K = np.exp(-gamma * pairwise_dists**2)
        return K

    def dpp_sampling(self, kernel_matrix, max_samples):
        """
        Perform a simple DPP sampling to select max_samples items.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(kernel_matrix)
        eigenvalues = np.flip(eigenvalues)
        eigenvectors = np.flip(eigenvectors, axis=1)

        # Select samples based on eigenvalues
        selected_samples = []
        cumulative_sum = 0
        for i, eigenvalue in enumerate(eigenvalues):
            if cumulative_sum + eigenvalue <= max_samples:
                cumulative_sum += eigenvalue
                selected_samples.append(i)
            if len(selected_samples) == max_samples:
                break

        selected_indices = np.where(
            np.isin(eigenvalues, eigenvalues[selected_samples])
        )[0]
        return selected_indices

    def update(self, minibatches, opt, sch, gamma, alpha):
        all_x = torch.cat([data[0].to(device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(device).long() for data in minibatches])

        # Compute the RBF kernel matrix
        features = self.featurizer(all_x)

        predicted = self.classifier(features)
        kernel_matrix = rbf_kernel(
            features.detach().cpu().numpy(), gamma=gamma
        )  # large gamma values --> narrow rbf kernel and vice versa
        diversity_loss = -torch.logdet(torch.from_numpy(kernel_matrix))

        self.dpp_sampling(kernel_matrix, max_samples=200)
        loss = F.cross_entropy(predicted, all_y)

        total_loss = alpha * loss + (1 - alpha) * diversity_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {"class": total_loss.item()}

    def predict(self, x):
        return self.network(x)
