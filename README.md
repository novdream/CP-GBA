
# CP-GBA: Cross-Paradigm Graph Backdoor Attacks with Promptable Subgraph Triggers

<table style="width:20%; border-collapse: collapse;">
  <tr>
    <th style="border: 1px solid black;text-align:center;"><a href="https://arxiv.org/abs/2510.22555">Paper</a></th>
  </tr>
</table>


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
The experiments are conducted on three public real-world datasets, i.e., Cora, Pubmed, Facebook and OGB-arxiv, which can be automatically downloaded to `./data` through torch-geometric API.

## Training and Testing

To train and test CP-GBA , run this command:

```train
bash train.sh
```

You can set the dataset parameter in  `train.sh` to achieve the evaluation of specific datasets. The specific parameters are elaborated in detail in the paper.

## Reference
If you find our code useful for your research, please consider citing our paper.
```
@misc{liu2026crossparadigmgraphbackdoorattacks,
      title={Cross-Paradigm Graph Backdoor Attacks with Promptable Subgraph Triggers}, 
      author={Dongyi Liu and Jiangtong Li},
      year={2026},
      eprint={2510.22555},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2510.22555}, 
}
```






