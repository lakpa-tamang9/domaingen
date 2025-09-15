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


def parse_seeds(s):
    if not s:
        return [0]
    return [int(x) for x in s.split(",")]


def get_args():
    parser = argparse.ArgumentParser(description="DG")
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--dataset", type=str, default="pacs", help="dataset name")
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
    parser.add_argument("--feat_mod", action="store_true", default=False)
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
        "--seeds",
        type=str,
        default="0",
        help="comma-separated seeds for multi-run stats",
    )
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
    args.seeds = parse_seeds(args.seeds)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    return args


args = get_args()


log_filename = f'logs/{args.dataset}_{args.algorithm}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger()


def compute_stats(values):
    arr = np.array(values, dtype=np.float64)
    mean = arr.mean()
    stderr = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, stderr


def train_one_epoch(args, model, train_loaders, optimizer, alpha, device=device):
    model.train()
    train_minibatches_iterator = zip(*train_loaders)

    last_loss, last_div_loss = 0.0, 0.0

    for _ in range(args.steps_per_epoch):
        minibatches = [(data) for data in next(train_minibatches_iterator)]
        x_all = torch.cat([data[0].to(device).float() for data in minibatches])
        y_all = torch.cat([data[1].to(device).long() for data in minibatches])

        feat = model.featurizer(x_all)  # [B, D]
        logits = model.classifier(feat)  # [B, C]

        if args.feat_mod:  # perform feature modulation with entropy
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * probs.log(), dim=1)
            entropy = (entropy - entropy.mean()) / (entropy.std() + 1e-6)
            feat = F.normalize(feat, dim=1)
            feat_weighted = feat * entropy.unsqueeze(1)
        else:
            feat_weighted = feat

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

        cls_loss = F.cross_entropy(logits, y_all)
        diversity_loss = -torch.logdet(K_t)
        total_loss = alpha * cls_loss + (1 - alpha) * diversity_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        last_loss = cls_loss.item()
        last_div_loss = diversity_loss.item()

    return last_loss, last_div_loss


@torch.no_grad()
def evaluate(eval_name_dict, model, eval_loaders):
    acc_record = {}
    for split in ["valid", "target"]:
        acc_record[split] = np.mean(
            np.array(
                [
                    modelopera.accuracy(model, eval_loaders[i])
                    for i in eval_name_dict[split]
                ]
            )
        )
    return acc_record


def run_training_for_seed(args, dataset, alpha, test_env, seed):
    """One full training run for a given seed. Returns best valid/target over epochs."""
    set_random_seed(seed)
    train_loaders, eval_loaders = get_img_dataloader_mod(args, dataset, [test_env])
    eval_name_dict = train_valid_target_eval_names(args, [test_env])

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm_pre = algorithm_class(args)
    algorithm = nn.DataParallel(algorithm_pre).to(device)
    opt = get_optimizer(algorithm_pre, args)

    best_valid, best_target = 0.0, 0.0
    last_cls, last_div = 0.0, 0.0

    for epoch in tqdm(
        range(args.max_epoch), desc=f"Seed {seed} | {dataset} env={test_env}"
    ):
        last_cls, last_div = train_one_epoch(
            args, algorithm.module, train_loaders, opt, alpha=alpha, device=device
        )
        acc_record = evaluate(eval_name_dict, algorithm.module, eval_loaders)

        # track the epoch-best target (and corresponding valid)
        if acc_record["target"] > best_target:
            best_target = acc_record["target"]
            best_valid = acc_record["valid"]

        if (epoch + 1) % max(1, args.checkpoint_freq) == 0:
            logger.info(
                f"[{dataset} env={test_env}] Seed {seed} | Epoch {epoch+1:03d} "
                f"| cls {last_cls:.4f} | dpp {last_div:.4f} "
                f"| valid {acc_record['valid']*100:.2f} | target {acc_record['target']*100:.2f} "
                f"| best_target {best_target*100:.2f}"
            )

    return best_valid, best_target


def main(args):
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

        for alpha in range(5, 6):  # for ablation run, change to range(0, 11)
            alpha = alpha / 10.0
            logger.info(f"=== Alpha: {alpha} ===")

            for test_env in range(domain_cnt):
                logger.info(
                    f"=== Dataset {dataset} | Target domain: {args.img_dataset[dataset][test_env]} ==="
                )

                # Run independent trainings for each seed
                run_valids, run_targets = [], []

                for seed in args.seeds:
                    best_valid, best_target = run_training_for_seed(
                        args, dataset, alpha, test_env, seed
                    )
                    run_valids.append(best_valid)
                    run_targets.append(best_target)
                    logger.info(
                        f"[{dataset} env={test_env}] Seed {seed} finished | best_valid={best_valid*100:.2f}, best_target={best_target*100:.2f}"
                    )

                # Aggregate stats across seeds (THIS is your mean ± SEM)
                v_mean, v_sem = compute_stats(run_valids)
                t_mean, t_sem = compute_stats(run_targets)

                logger.info(
                    f"[{dataset} env={test_env}] "
                    f"VALID: {v_mean*100:.2f} ± {v_sem*100:.2f} | "
                    f"TARGET: {t_mean*100:.2f} ± {t_sem*100:.2f}  (mean ± SEM over seeds {args.seeds})"
                )


if __name__ == "__main__":
    main(args)
    logger.info("Training completed successfully.")
    logger.info("All done!")
