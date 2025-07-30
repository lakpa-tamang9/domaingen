# Domain Generation with Diversity Sampling

## Datasets:

- Download datasets and put it in `./data` folder. These dataset PACS, VLCS, OfficeHome, Terra-Incognita, and DomainNet can be download from [here](https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py)

## Usage

Run `python dpp.py. This performs following.

- Prepares dataset
- Trains the model using all the dataset and their respective domains and logs the results in `logs/dpp_train_xxxxx.log` file. The model is validated and the best accuracy on target domain is stored along training.
