# coding=utf-8
import numpy as np
import torch.nn.functional as F
import torch
from datautil.util import random_pairs_of_minibatches
from alg.algs.ERM import ERM
from sklearn.metrics.pairwise import rbf_kernel


class Mixup_DPP(ERM):
    def __init__(self, args):
        super(Mixup_DPP, self).__init__(args)
        self.args = args

    def update_org(self, minibatches, opt, sch):
        objective = 0

        for (xi, yi, di), (xj, yj, dj) in random_pairs_of_minibatches(
            self.args, minibatches
        ):
            lam = np.random.beta(
                self.args.mixupalpha, self.args.mixupalpha
            )  # generate sample from beta distribution on interval [0, 1]

            x = (lam * xi + (1 - lam) * xj).cuda().float()

            predictions = self.predict(x)

            objective += lam * F.cross_entropy(
                predictions, yi.cuda().long()
            )  # loss compute between model's prediction and label of first input yi
            objective += (1 - lam) * F.cross_entropy(
                predictions, yj.cuda().long()
            )  # loss compute between model's prediction and label of first input yj

        objective /= len(minibatches)

        opt.zero_grad()
        objective.backward()
        opt.step()
        if sch:
            sch.step()
        return {"class": objective.item()}

    def update(self, minibatches, opt, sch):
        objective = 0
        diversity_loss = 0

        for (xi, yi, di), (xj, yj, dj) in random_pairs_of_minibatches(
            self.args, minibatches
        ):
            lam = np.random.beta(
                self.args.mixupalpha, self.args.mixupalpha
            )  # generate sample from beta distribution on interval [0, 1]

            x = (lam * xi + (1 - lam) * xj).cuda().float()

            # Predict
            predictions = self.predict(x)

            # Compute Cross-Entropy Loss
            objective += lam * F.cross_entropy(predictions, yi.cuda().long())
            objective += (1 - lam) * F.cross_entropy(predictions, yj.cuda().long())

            # Extract features from mixed-up data for DPP loss
            features = self.featurizer(
                x
            )  # Assume `extract_features` is your method to get features from x

            # Compute the RBF Kernel Matrix for DPP loss
            kernel_matrix = rbf_kernel(
                features.detach().cpu().numpy(), gamma=0.5
            )  # `gamma` is a hyperparameter that you might want to tune
            diversity_loss += -torch.logdet(
                torch.from_numpy(kernel_matrix).float().cuda()
            )

            # dpp_loss + = diversity_loss

        # Average the losses
        objective /= len(minibatches)
        diversity_loss /= len(minibatches)

        # Combine cross-entropy and DPP loss
        total_loss = objective + 0.5 * diversity_loss

        # Optimization step
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if sch:
            sch.step()

        return {"class": total_loss.item()}
