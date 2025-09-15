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

    def update(self, minibatches, opt, sch, alpha):
        all_x = torch.cat([data[0].to(device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(device).long() for data in minibatches])

        # Compute the RBF kernel matrix
        features = self.featurizer(all_x)

        logits = self.classifier(features)
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * probs.log(), dim=1)
        entropy = (entropy - entropy.mean()) / (entropy.std() + 1e-6)
        feat = F.normalize(features, dim=1)
        feat_weighted = feat * entropy.unsqueeze(1)

        # median heuristic for gamma on weighted features
        with torch.no_grad():
            dist_sq = (
                (feat_weighted.unsqueeze(0) - feat_weighted.unsqueeze(1)) ** 2
            ).sum(2)
            gamma = 1.0 / (dist_sq.median() + 1e-8)

        # RBF kernel via sklearn (CPU), then bring back as torch on device
        K = rbf_kernel(feat_weighted.detach().cpu().numpy(), gamma=gamma.item())
        K = K / (np.trace(K) + 1e-6)
        K += np.eye(K.shape[0]) * 1e-1
        K_t = torch.tensor(K, device=device, dtype=feat.dtype)

        cls_loss = F.cross_entropy(logits, all_y)
        diversity_loss = -torch.logdet(K_t)
        total_loss = alpha * cls_loss + (1 - alpha) * diversity_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if sch:
            sch.step()
        return {"class": total_loss.item()}

    def predict(self, x):
        return self.network(x)
