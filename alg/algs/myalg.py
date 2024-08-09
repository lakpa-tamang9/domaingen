import torch
import torch.nn as nn
import torch.nn.functional as F
from alg.algs.base import Algorithm
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AAE(Algorithm):
    def __init__(self, args, hidden_layer=1000, activation="softmax"):

        super(AAE, self).__init__(args)

        self.hidden_layer = hidden_layer
        self.encoder = Encoder(args.input_shape, hidden_layer)
        self.decoder = Decoder(args.input_shape, hidden_layer)
        self.adversarial = Adversarial(hidden_layer)
        self.classifier = Classifier(hidden_layer, args.num_classes, activation)
        self.latent_dims = 120
        self.args = args

    def lock_model(self, model):
        for name, param in model.named_parameters():
            if "Adv" not in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return model

    def update(self, minibatches, opt, sch, eps=1e-15):
        all_x = torch.cat([data[0].to(device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(device).long() for data in minibatches])

        all_x_f, all_z = self.encoder(all_x)
        all_recon_X = self.decoder(all_z)

        all_preds = self.classifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)
        recon_loss = F.mse_loss(all_x_f + eps, all_recon_X + eps)

        z_real_gauss = Variable(
            torch.randn(all_x.size()[0], self.hidden_layer) * 5.0
        ).to(device)
        D_real_gauss = self.adversarial(z_real_gauss)

        z_fake_gauss = all_z
        D_fake_gauss = self.adversarial(z_fake_gauss)

        adv_loss = -torch.mean(
            torch.log(D_real_gauss + eps) + torch.log(1 - D_fake_gauss + eps)
        )

        loss = classifier_loss + adv_loss + recon_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {
            "total": loss.item(),
            "class": classifier_loss.item(),
            "adv": adv_loss.item(),
            "recon": recon_loss.item(),
        }

    def predict(self, x):
        _, inv_feats = self.encoder(x)
        return self.classifier(inv_feats)


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_shape, hidden_layer):
        super(Encoder, self).__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Linear(input_shape, hidden_layer)

    def forward(self, x):
        x_f = self.flatten(x)
        x = self.encoder(x_f)
        return x_f, x


# Decoder
class Decoder(nn.Module):
    def __init__(self, input_shape, hidden_layer):
        super(Decoder, self).__init__()
        self.hidden_layer = hidden_layer
        self.dropout = nn.Dropout(0.25)
        self.decoder = nn.Linear(hidden_layer, input_shape)

    def forward(self, x):
        x = self.dropout(x)
        x = self.decoder(x)
        return x


# Discriminator
class Adversarial(nn.Module):
    def __init__(self, hidden_layer):
        super(Adversarial, self).__init__()
        self.hidden_layer = hidden_layer
        self.fc1 = nn.Linear(hidden_layer, hidden_layer)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Classifier(nn.Module):
    def __init__(self, hidden_layer, num_classes, activation):
        super(Classifier, self).__init__()
        self.hidden_layer = hidden_layer
        self.num_classes = num_classes
        self.activation = activation

        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(hidden_layer, hidden_layer)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer, num_classes)

        if activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x
