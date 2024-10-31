# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM
from sklearn.metrics.pairwise import rbf_kernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CORAL_DPP(ERM):
    def __init__(self, args):
        super(CORAL_DPP, self).__init__(args)
        self.args = args
        self.kernel_type = "mean_cov"

    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def update(self, minibatches, opt, sch):
        objective = 0
        penalty = 0
        nmb = len(minibatches)
        features = [self.featurizer(data[0].cuda().float()) for data in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [data[1].cuda().long() for data in minibatches]

        all_x = torch.cat([data[0].to(device).float() for data in minibatches])
        features = self.featurizer(all_x)

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.coral(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        kernel_matrix = rbf_kernel(
            features.detach().cpu().numpy(), gamma=0.5
        )  # large gamma values --> narrow rbf kernel and vice versa

        diversity_loss = -torch.logdet(torch.from_numpy(kernel_matrix))

        objective += diversity_loss

        opt.zero_grad()
        (objective + (self.args.mmd_gamma * penalty)).backward()
        opt.step()
        if sch:
            sch.step()
        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {
            "class": objective.item(),
            "coral": penalty,
            "total": (objective.item() + (self.args.mmd_gamma * penalty)),
        }
