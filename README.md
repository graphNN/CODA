Class-Homophilic-based Data Augmentation for Improving Graph Neural Networks
====
This repository contains the source code for this paper:

## Requirements

This code package was developed and tested with Python 3.7.6. Make sure all dependencies specified in the ```requirements.txt``` file are satisfied before running the model. This can be achieved by
```
pip install -r requirements.txt
```

## Usage
The scripts for hyperparameter search with Optuna are ```optuna_[method].py```.

All the parameters are included in this paper. 
Results can be reproduced with the scripts ```CODA_train.py```, and parameters need to be modified manually. 
For example, to reproduce the result of graph-CODA-GCN on Cora, you need modify parameters:
```
args.dataset='cora'
args.model='GCN'
args.m=49
args.T0=2.9
args.T1=0.5
args.self_dropout0=0.5
args.self_dropout1=0.6
args.prior_model='grand'
```

## Data
Citation networks: Cora, Citeseer, Pubmed
WebKB: Wisconsin, Texas, Cornell

## Cite
If you find this repository useful in your research, please cite our paper:

```

```

