# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm
from sklearn.metrics.pairwise import rbf_kernel

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

        self.network = nn.Sequential(self.featurizer, self.classifier)

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].to(device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(device).long() for data in minibatches])

        # Compute the RBF kernel matrix
        features = self.featurizer(all_x).detach().cpu().numpy()
        # kernel_matrix = rbf_kernel(features, gamma=0.1)
        kernel_matrix = rbf_kernel(
            all_x, gamma=0.5
        )  # large gamma values --> narrow rbf kernel and vice versa

        diversity_loss = -torch.logdet(torch.from_numpy(kernel_matrix))

        loss = F.cross_entropy(self.predict(all_x), all_y)
        total_loss = 0.5 * loss + 0.5 * diversity_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {"class": total_loss.item()}

    def predict(self, x):
        return self.network(x)
