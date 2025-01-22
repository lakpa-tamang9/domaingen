# Domain Generation with Diversity Sampling

## How to Run:

- Download datasets and put it in `./data` folder. These dataset PACS, VLCS, OfficeHome, Terra-Incognita, and DomainNet can be download from [here](https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py)
- Run `train.py`
  - Prepares dataset
  - Trains model on different methods: original implementation and DPP sampling aided implementation
  - Results are stored as text files.
