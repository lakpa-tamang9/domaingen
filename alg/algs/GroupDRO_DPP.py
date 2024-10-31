# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM
from sklearn.metrics.pairwise import rbf_kernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GroupDRO_DPP(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, args):
        super(GroupDRO_DPP, self).__init__(args)
        self.register_buffer("q", torch.Tensor())
        self.args = args

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].to(device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(device).long() for data in minibatches])
        if not len(self.q):
            self.q = torch.ones(len(minibatches)).cuda()

        losses = torch.zeros(len(minibatches)).cuda()

        for m in range(len(minibatches)):
            x, y = minibatches[m][0].cuda().float(), minibatches[m][1].cuda().long()
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.args.groupdro_eta * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)
        # Compute the RBF kernel matrix
        features = self.featurizer(all_x)
        kernel_matrix = rbf_kernel(
            features.detach().cpu().numpy(), gamma=0.5
        )  # large gamma values --> narrow rbf kernel and vice versa

        diversity_loss = -torch.logdet(torch.from_numpy(kernel_matrix))
        total_loss = loss + diversity_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        if sch:
            sch.step()

        return {"group": total_loss.item()}
