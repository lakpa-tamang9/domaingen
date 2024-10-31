# coding=utf-8
import torch
import torch.nn.functional as F

from alg.algs.ERM import ERM
from sklearn.metrics.pairwise import rbf_kernel


class MMDDPP(ERM):
    def __init__(self, args):
        super(MMDDPP, self).__init__(args)
        self.args = args
        self.kernel_type = "gaussian"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(
            x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
        ).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def update(self, minibatches, opt, sch):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        # Extract features and classifications for each minibatch
        features = [self.featurizer(data[0].cuda().float()) for data in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [data[1].cuda().long() for data in minibatches]

        # Compute classification loss and MMD penalty
        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        # Compute average classification loss and MMD penalty
        objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        # Compute diversity loss using the RBF kernel matrix
        combined_features = torch.cat(
            features, dim=0
        )  # Combine all features into a single tensor
        kernel_matrix = rbf_kernel(combined_features.detach().cpu().numpy(), gamma=0.5)
        diversity_loss = -torch.logdet(torch.from_numpy(kernel_matrix))

        # Combine the classification loss, MMD penalty, and diversity loss
        total_loss = (
            objective + (self.args.mmd_gamma * penalty) + (0.5 * diversity_loss)
        )

        # Perform backward pass and optimizer step
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if sch:
            sch.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {
            "class": objective.item(),
            "mmd": penalty,
            "diversity_loss": diversity_loss.item(),
            "total": total_loss.item(),
        }
