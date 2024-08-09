# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader

import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from datautil.mydataloader import InfiniteDataLoader


def get_img_dataloader(args):
    rate = 0.2
    trdatalist, tedatalist = [], []

    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        if i in args.test_envs:
            tedatalist.append(
                ImageDataset(
                    args.dataset,
                    args.task,
                    args.data_dir,
                    names[i],
                    i,
                    transform=imgutil.image_test(args.dataset),
                    test_envs=args.test_envs,
                )
            )
        else:
            tmpdatay = ImageDataset(
                args.dataset,
                args.task,
                args.data_dir,
                names[i],
                i,
                transform=imgutil.image_train(args.dataset),
                test_envs=args.test_envs,
            ).labels
            l = len(tmpdatay)
            if args.split_style == "strat":
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1 - rate, random_state=args.seed
                )
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l * rate)
                indextr, indexte = indexall[:-ted], indexall[-ted:]

            trdatalist.append(
                ImageDataset(
                    args.dataset,
                    args.task,
                    args.data_dir,
                    names[i],
                    i,
                    transform=imgutil.image_train(args.dataset),
                    indices=indextr,
                    test_envs=args.test_envs,
                )
            )
            tedatalist.append(
                ImageDataset(
                    args.dataset,
                    args.task,
                    args.data_dir,
                    names[i],
                    i,
                    transform=imgutil.image_test(args.dataset),
                    indices=indexte,
                    test_envs=args.test_envs,
                )
            )

    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=None,
            batch_size=args.batch_size,
            num_workers=args.N_WORKERS,
        )
        for env in trdatalist
    ]

    eval_loaders = [
        DataLoader(
            dataset=env,
            batch_size=64,
            num_workers=args.N_WORKERS,
            drop_last=False,
            shuffle=False,
        )
        for env in trdatalist + tedatalist
    ]

    return train_loaders, eval_loaders


def get_img_dataloader_mod2(args, test_envs):
    rate = 0.2
    trdatalist, tedatalist = [], []

    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        if i in test_envs:
            tedatalist.append(
                ImageDataset(
                    args.dataset,
                    args.task,
                    args.data_dir,
                    names[i],
                    i,
                    transform=imgutil.image_test(args.dataset),
                    test_envs=test_envs,
                )
            )
        else:
            tmpdatay = ImageDataset(
                args.dataset,
                args.task,
                args.data_dir,
                names[i],
                i,
                transform=imgutil.image_train(args.dataset),
                test_envs=test_envs,
            ).labels
            l = len(tmpdatay)
            if args.split_style == "strat":
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1 - rate, random_state=args.seed
                )
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l * rate)
                indextr, indexte = indexall[:-ted], indexall[-ted:]

            trdatalist.append(
                ImageDataset(
                    args.dataset,
                    args.task,
                    args.data_dir,
                    names[i],
                    i,
                    transform=imgutil.image_train(args.dataset),
                    indices=indextr,
                    test_envs=test_envs,
                )
            )
            tedatalist.append(
                ImageDataset(
                    args.dataset,
                    args.task,
                    args.data_dir,
                    names[i],
                    i,
                    transform=imgutil.image_test(args.dataset),
                    indices=indexte,
                    test_envs=test_envs,
                )
            )

    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=None,
            batch_size=args.batch_size,
            num_workers=args.N_WORKERS,
        )
        for env in trdatalist
    ]

    eval_loaders = [
        DataLoader(
            dataset=env,
            batch_size=64,
            num_workers=args.N_WORKERS,
            drop_last=False,
            shuffle=False,
        )
        for env in trdatalist + tedatalist
    ]

    return train_loaders, eval_loaders


def get_img_dataloader_mod(args, dset, test_env):
    rate = 0.2
    trdatalist, tedatalist = [], []

    names = args.img_dataset[dset]
    print(names)
    args.domain_num = len(names)
    for i in range(len(names)):
        if i in test_env:
            # Create test data array
            tedatalist.append(
                ImageDataset(
                    dset,
                    args.task,
                    args.data_dir,
                    names[i],
                    i,
                    transform=imgutil.image_test(dset),
                    test_envs=test_env,
                )
            )
        else:
            tmpdatay = ImageDataset(
                dset,
                args.task,
                args.data_dir,
                names[i],
                i,
                transform=imgutil.image_train(dset),
                test_envs=test_env,
            ).labels
            l = len(tmpdatay)
            if args.split_style == "strat":
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1 - rate, random_state=args.seed
                )

                """
                Here's a breakdown of how StratifiedShuffleSplit works:

                    Stratification: The split is stratified, meaning that it attempts to maintain the percentage of samples of each class.
                    Shuffling: It shuffles the data before splitting into batches.
                    Batch Allocation: No data point will appear in more than one test set split, and similarly, no train-test overlap is allowed.
                """
                stsplit.get_n_splits(lslist, tmpdatay)
                splits = stsplit.split(lslist, tmpdatay)
                indextr, indexte = next(splits)
            else:
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l * rate)
                indextr, indexte = indexall[:-ted], indexall[-ted:]

            trdatalist.append(
                ImageDataset(
                    dset,
                    args.task,
                    args.data_dir,
                    names[i],
                    i,
                    transform=imgutil.image_train(dset),
                    indices=indextr,
                    test_envs=test_env,
                )
            )
            tedatalist.append(
                ImageDataset(
                    dset,
                    args.task,
                    args.data_dir,
                    names[i],
                    i,
                    transform=imgutil.image_test(dset),
                    indices=indexte,
                    test_envs=test_env,
                )
            )

    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=None,
            batch_size=args.batch_size,
            num_workers=args.N_WORKERS,
        )
        for env in trdatalist
    ]

    eval_loaders = [
        DataLoader(
            dataset=env,
            batch_size=64,
            num_workers=args.N_WORKERS,
            drop_last=False,
            shuffle=False,
        )
        for env in trdatalist + tedatalist
    ]

    return train_loaders, eval_loaders
