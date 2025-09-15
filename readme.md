# Uncertainty-guided Diversity Sampling Regularization for Domain Generalization

## Datasets:

- Download datasets and put it in `./data` folder. These dataset PACS, VLCS, OfficeHome, Terra-Incognita, and DomainNet can be download from [here](https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py)

## Usage

To train, do following:

- With feature modulation:
  ```
  python train.py --feat_mod
  ```
- Without feature modulation:
  ```
  python train.py
  ```

## Training and Evaluation Overview

Trains the model with three independent runs on different seeds. The outputs are logged as following:

```
logs/dpp_train_xxxxx.log
```

Trains the model using all the dataset and their respective domains and logs the results in `logs/{dataset}_{algorithm}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'` file. Default algorithm is ERM. It can be changed to other algorithms that are present inside `alg/algs/...`
