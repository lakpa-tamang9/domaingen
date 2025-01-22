# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm
from sklearn.metrics.pairwise import rbf_kernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, args):
        super(ERM, self).__init__(args)
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier
        )

        self.network = nn.Sequential(self.featurizer, self.classifier)

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].to(device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(device).long() for data in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {"class": loss.item()}

    def predict(self, x):
        return self.network(x)


class MBDG_Base(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MBDG_Base, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.G = load_munit_model(
            self.hparams["mbdg_model_path"], self.hparams["mbdg_config_path"]
        )

    @staticmethod
    def kl_div(dist1, dist2):
        return F.kl_div(torch.log(dist2), dist1, reduction="batchmean")

    def predict(self, x):
        return self.network(x)

    @torch.no_grad()
    def generate_images(self, images):
        delta = (
            torch.randn(images.size(0), self.G.delta_dim, 1, 1)
            .cuda()
            .requires_grad_(False)
        )
        return self.G(images, delta)

    def calc_dist_reg(self, x, clean_output):
        mb_images = self.generate_images(x)
        mb_output = F.softmax(self.predict(mb_images), dim=1)
        return self.kl_div(F.softmax(clean_output, dim=1), mb_output)

    @staticmethod
    def relu(x):
        return x if x > 0 else torch.tensor(0).cuda()


class MBDG(MBDG_Base):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MBDG, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.dual_var = torch.tensor(1.0).cuda().requires_grad_(False)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        clean_output = self.predict(all_x)
        clean_loss = F.cross_entropy(clean_output, all_y)
        dist_reg = self.calc_dist_reg(all_x, clean_output)

        loss = clean_loss + self.dual_var * dist_reg

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        const_unsat = dist_reg.detach() - self.hparams["mbdg_gamma"]
        self.dual_var = self.relu(
            self.dual_var + self.hparams["mbdg_dual_step_size"] * const_unsat
        )

        return {
            "loss": loss.item(),
            "dist_reg": dist_reg.item(),
            "dual_var": self.dual_var.item(),
        }


class MBDGDPP(MBDG_Base):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MBDG, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.dual_var = torch.tensor(1.0).cuda().requires_grad_(False)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # Compute the RBF kernel matrix
        features = self.featurizer(all_x)
        kernel_matrix = rbf_kernel(
            features.detach().cpu().numpy(), gamma=0.5
        )  # large gamma values --> narrow rbf kernel and vice versa

        diversity_loss = -torch.logdet(torch.from_numpy(kernel_matrix))

        clean_output = self.predict(all_x)
        clean_loss = F.cross_entropy(clean_output, all_y)
        dist_reg = self.calc_dist_reg(all_x, clean_output)

        loss = clean_loss + self.dual_var * dist_reg + 0.5 * diversity_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        const_unsat = dist_reg.detach() - self.hparams["mbdg_gamma"]
        self.dual_var = self.relu(
            self.dual_var + self.hparams["mbdg_dual_step_size"] * const_unsat
        )

        return {"loss": loss.item()}
