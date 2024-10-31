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
            args.num_classes, self.featurizer.in_features, args.classifier
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

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].to(device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(device).long() for data in minibatches])

        # Compute the RBF kernel matrix
        features = self.featurizer(all_x)
        predicted = self.classifier(features)
        kernel_matrix = rbf_kernel(
            features.detach().cpu().numpy(), gamma=0.5
        )  # large gamma values --> narrow rbf kernel and vice versa

        diversity_loss = -torch.logdet(torch.from_numpy(kernel_matrix))
        loss = F.cross_entropy(predicted, all_y)

        total_loss = loss + 0.5 * diversity_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        if sch:
            sch.step()
        return features, all_y, {"class": total_loss.item()}

    def update_mod1(self, minibatches, opt, sch):
        """
        1. Use DPP sampling to select certain percentage of features (eg: 50% of total batch size)
        2. Do prediction of selected features and compute loss
        3. Compute diversity loss through negative log determinant of the selected kernel matrix
        4. Compute CE loss of all samples
        5. Combine all losses
        """
        all_x = torch.cat([data[0].to(device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(device).long() for data in minibatches])

        # Compute the RBF kernel matrix
        features = self.featurizer(all_x)
        predicted = self.classifier(features)

        kernel_matrix = self.custom_kernel(features.detach().cpu().numpy())

        selected_indices = self.dpp_sampling(kernel_matrix, 0.5 * len(all_x))

        selected_features = features[selected_indices]

        # selected_features_tensor = torch.from_numpy(selected_features).to(device)
        selected_predictions = self.classifier(selected_features)
        selected_labels = all_y[selected_indices]
        selected_loss = F.cross_entropy(selected_predictions, selected_labels)

        selected_kernel_matrix = kernel_matrix[
            np.ix_(selected_indices, selected_indices)
        ]

        diversity_loss = -torch.logdet(torch.from_numpy(selected_kernel_matrix))

        # diversity_loss = -torch.logdet(torch.from_numpy(kernel_matrix))

        loss = F.cross_entropy(predicted, all_y)

        total_loss = loss + 0.5 * diversity_loss + 0.5 * selected_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {"class": total_loss.item()}

    def predict(self, x):
        return self.network(x)
