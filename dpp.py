import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import sys
from datetime import datetime
import os
from alg.opt import *
from alg import alg, modelopera
from utils.util import (
    set_random_seed,
    train_valid_target_eval_names,
    img_param_init,
)
from datautil.getdataloader import (
    get_img_dataloader_mod,
)
from tqdm import tqdm
import argparse
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


def get_args():
    parser = argparse.ArgumentParser(description="DG")
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument(
        "--steps_per_epoch", type=int, default=100, help="steps per epoch"
    )
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam hyper-param")
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument(
        "--checkpoint_freq", type=int, default=3, help="Checkpoint every N epoch"
    )
    parser.add_argument(
        "--classifier", type=str, default="wn", choices=["linear", "wn"]
    )
    parser.add_argument("--data_file", type=str, default="", help="root_dir")
    parser.add_argument("--data_dir", type=str, default="", help="data dir")
    parser.add_argument(
        "--dis_hidden", type=int, default=256, help="dis hidden dimension"
    )
    parser.add_argument(
        "--disttype",
        type=str,
        default="2-norm",
        choices=["1-norm", "2-norm", "cos", "norm-2-norm", "norm-1-norm"],
    )
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument(
        "--inner_lr", type=float, default=1e-2, help="learning rate used in MLDG"
    )
    parser.add_argument(
        "--lam", type=float, default=1, help="tradeoff hyperparameter used in VREx"
    )
    parser.add_argument("--layer", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.75, help="for sgd")
    parser.add_argument(
        "--lr_decay1", type=float, default=1.0, help="for pretrained featurizer"
    )
    parser.add_argument(
        "--lr_decay2",
        type=float,
        default=1.0,
        help="inital learning rate decay of network",
    )
    parser.add_argument("--lr_gamma", type=float, default=0.0003, help="for optimizer")
    parser.add_argument("--max_epoch", type=int, default=100, help="max iterations")
    parser.add_argument(
        "--mixupalpha", type=float, default=0.2, help="mixup hyper-param"
    )

    parser.add_argument("--momentum", type=float, default=0.9, help="for optimizer")
    parser.add_argument(
        "--net",
        type=str,
        default="resnet50",
        help="featurizer: vgg16, resnet18, resnet50, resnet101,DTNBase",
    )
    parser.add_argument("--N_WORKERS", type=int, default=4)
    parser.add_argument(
        "--rsc_f_drop_factor", type=float, default=1 / 3, help="rsc hyper-param"
    )
    parser.add_argument(
        "--rsc_b_drop_factor", type=float, default=1 / 3, help="rsc hyper-param"
    )
    parser.add_argument("--save_model_every_checkpoint", action="store_true")
    parser.add_argument("--schuse", action="store_true")
    parser.add_argument("--schusech", type=str, default="cos")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split_style",
        type=str,
        default="strat",
        help="the style to split the train and eval datasets",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="img_dg",
        choices=["img_dg"],
        help="now only support image tasks",
    )
    parser.add_argument("--tau", type=float, default=1, help="andmask tau")
    parser.add_argument(
        "--test_envs",
        type=int,
        nargs="+",
        default=[0],
        help="target domains, test domain (other domains will be used for training)",
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    args = parser.parse_args()
    args.steps_per_epoch = 100
    args.data_dir = args.data_file + args.data_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    return args


# Create unique log filename with timestamp
log_filename = f'logs/dpp_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout),  # log to console
    ],
)

logger = logging.getLogger()


def train(
    args,
    model,
    train_loaders,
    optimizer,
    alpha,
    device=device,
):

    model.train()
    train_minibatches_iterator = zip(*train_loaders)

    for step in range(args.steps_per_epoch):
        minibatches = [(data) for data in next(train_minibatches_iterator)]
        x_all = torch.cat([data[0].to(device).float() for data in minibatches])
        y_all = torch.cat([data[1].to(device).long() for data in minibatches])
        d_all = torch.cat(
            [
                torch.full((data[0].size(0),), idx, dtype=torch.long).to(device)
                for idx, data in enumerate(minibatches)
            ]
        )  # domain index per sample

        feat = model.featurizer(x_all)  # [B, 512]
        logits = model.classifier(feat)  # task prediction

        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * probs.log(), dim=1)
        entropy = (entropy - entropy.mean()) / (entropy.std() + 1e-6)
        feat = F.normalize(feat, dim=1)

        feat_weighted = feat * entropy.unsqueeze(1)  # [B, D] × [B, 1]

        with torch.no_grad():
            dist_sq = (
                (feat_weighted.unsqueeze(0) - feat_weighted.unsqueeze(1)) ** 2
            ).sum(
                2
            )  # same like euclidean distance
            gamma = 1.0 / (dist_sq.median() + 1e-8)

        kernel_matrix = rbf_kernel(
            feat_weighted.detach().cpu().numpy(), gamma=gamma.item()
        )

        kernel_matrix = kernel_matrix / (
            np.trace(kernel_matrix) + 1e-6
        )  # Scale to avoid large logdet
        kernel_matrix += np.eye(kernel_matrix.shape[0]) * 1e-1

        loss = F.cross_entropy(logits, y_all)

        diversity_loss = -torch.logdet(torch.from_numpy(kernel_matrix))
        total_loss = alpha * loss + (1 - alpha) * diversity_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return loss.item(), diversity_loss.item()


def test(eval_name_dict, model, eval_loaders):
    acc_record = {}
    acc_type_list = ["valid", "target"]
    for item in acc_type_list:
        acc_record[item] = np.mean(
            np.array(
                [
                    modelopera.accuracy(model, eval_loaders[i])
                    for i in eval_name_dict[item]
                ]
            )
        )

    return acc_record


def set_random_seed(seed):
    import random, numpy as np, torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_stats(values):
    """
    Compute mean and standard error of the mean (SEM) for a list of numbers.
    """
    arr = np.array(values)
    mean = arr.mean()
    stderr = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, stderr


def main():
    args = get_args()
    dataset_info = {
        "PACS": 4,
        "VLCS": 4,
        "OfficeHome": 4,
        "TerraIncognita": 4,
        "DomainNet": 6,
    }
    for dataset, domain_cnt in dataset_info.items():
        args = img_param_init(args, dataset=dataset)
        args.data_dir = f"data/{dataset}/"
        for test_env in range(domain_cnt):
            logger.info(f"Target dataset set to {args.img_dataset[dataset][test_env]}")
            train_loaders, eval_loaders = get_img_dataloader_mod(
                args, dataset, [test_env]
            )
            eval_name_dict = train_valid_target_eval_names(args, [test_env])

            algorithm_class = alg.get_algorithm_class(args.algorithm)

            algorithm = algorithm_class(args).to(device)

            opt = get_optimizer(algorithm, args)

            best_valid_acc, best_target_acc, target_acc = 0, 0, 0
            for epoch in tqdm(range(args.max_epoch)):
                target_trial_accs = []
                valid_trial_accs = []
                # train_trial_accs = []
                for trial in range(3):
                    set_random_seed(42 + epoch + trial)
                    loss, div_loss = train(
                        args,
                        algorithm,
                        train_loaders,
                        opt,
                        alpha=0.5,
                    )

                    acc_record = test(
                        eval_name_dict=eval_name_dict,
                        model=algorithm,
                        eval_loaders=eval_loaders,
                    )
                    # Update the accuracies
                    if acc_record["target"] > target_acc:
                        best_valid_acc = acc_record["valid"]
                        target_acc = acc_record["target"]

                    target_trial_accs.append(target_acc)
                    valid_trial_accs.append(acc_record["valid"])

                target_mean, target_stderr = compute_stats(target_trial_accs)
                valid_mean, valid_stderr = compute_stats(valid_trial_accs)

                logger.info(
                    f"Epoch {epoch+1:02d} | class_loss {loss:.4f} | dpp_loss {div_loss:.4f} "
                    f"| Valid Acc: {valid_mean*100:.2f} ± {valid_stderr*100:.2f} "
                    f"| Target Acc: {target_mean*100:.2f} ± {target_stderr*100:.2f}"
                )


if __name__ == "__main__":
    main()
    logger.info("Training completed successfully.")
    logger.info("All done!")
