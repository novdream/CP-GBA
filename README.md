
# CP-GBA: Cross-Paradigm Graph Backdoor Attacks with Promptable Subgraph Triggers


## Introduction

CP-GBA is an innovative framework designed to address the critical issue of crafting transferable graph backdoor attacks across diverse learning paradigms. This framework enhances the adaptability of graph backdoor attacks, ensuring they remain effective across various graph learning models.


**Components**
- **Condensed Subgraph Triggers:** Constructs a set of diverse triggers that maintain the intrinsic structural properties of the data.
- **Graph Prompt Learning for Trigger Training:** Employs GPL to train triggers, leveraging its theoretical transferability properties, to ensure the triggers are effective across different learning paradigms.

**Contributions:**
-  **Problem Exploration:** We study a novel backdoor attack problem aimed at generalizing attacks across different graph learning paradigms.
-  **Method Development:** Based on recent research on GPL, we explore using GPL to train backdoor triggers and verify its effectiveness in generalizing triggers across various learning paradigms.

## Overview

*  `utils.py`: The functions to select nodes, load and split data.
*  `CPGBA.py`: This file contains the model of CP-GBA.
*  `construct.py`: This file contains target model info.
*  `surrogate_models`: The framework of baseline backdoor attack.
*  `pre_trained_gnn`: The framework of baseline backdoor attack. 
*  `train.py`: The program to run our CP-GBA attack.


## Requirements

To install requirements:

```setup
conda create -n CP-GBA python=3.9
conda activate CP-GBA

# Install scikit-learn-extra
pip install scikit-learn-extra==0.3.0

# Install Pytorch with CUDA support
# Please adjust the CUDA version if necessary
conda install pytorch==2.21 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install torch_geometric
pip install torch_geometric==2.5.0

# Install ogb
pip install ogb==1.3.6

# Install additional dependencies
pip install pandas torchmetrics Deprecated
```
## Data
The experiments are conducted on three public real-world datasets, i.e., Cora, Pubmed, Facebook, which can be automatically downloaded to `./data` through torch-geometric API.

## Training and Testing

To train and test CP-GBA , run this command:

```train
bash train.sh
```

You can set the dataset parameter in  `train.sh` to achieve the evaluation of specific datasets. The specific parameters are elaborated in detail in the paper.

## Results

Our model achieves the following performance on(ASR | CA) :

|Dataset   | Method | GTA | UGBA | DPGBA |CP-GBA|
| -------- |------- | --- | ---- |------ | ------ |
|          |    GSL | 0.75 \| 0.82 | 0.76 \| 0.82 | 0.78 \| 0.81 | 0.97 \| 0.81 |
|   Cora   |    GCL | 0.25 \| 0.69 | 0.51 \| 0.70 | 0.09 \| 0.71 | 0.91 \| 0.76 |
|          |    GPL | 0.51 \| 0.21 | 0.63 \| 0.26 | 0.46 \| 0.29 | 0.99 \| 0.34 |
|          |    GSL | 0.79 \| 0.87 | 0.79 \| 0.86 | 0.66 \| 0.87 | 0.96 \| 0.84 |
|  Pubmed  |    GCL | 1.00 \| 0.20 | 0.67 \| 0.84 | 0.23 \| 0.84 | 0.93 \| 0.84 |
|          |    GPL | 0.54 \| 0.39 | 0.65 \| 0.50 | 0.82 \| 0.45 | 1.00 \| 0.44 |
|          |    GSL | 0.68 \| 0.88 | 0.80 \| 0.88 | 0.80 \| 0.88 | 0.92 \| 0.85 |
| Facebook |    GCL | 0.23 \| 0.83 | 0.84 \| 0.80 | 0.27 \| 0.78 | 0.92 \| 0.79 |
|          |    GPL | 0.33 \| 0.31 | 0.30 \| 0.34 | 0.36 \| 0.33 | 0.99 \| 0.39 |
|          |    GSL | 0.68 \| 0.88 | 0.80 \| 0.88 | 0.80 \| 0.88 | 0.92 \| 0.85 |
| OGB-arxiv|    GCL | 0.23 \| 0.83 | 0.84 \| 0.80 | 0.27 \| 0.78 | 0.92 \| 0.79 |
|          |    GPL | 0.33 \| 0.31 | 0.30 \| 0.34 | 0.36 \| 0.33 | 0.99 \| 0.39 |









