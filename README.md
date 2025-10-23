# Mamba2Diff

Official code for "Mamba2Diff: An Enhanced Diffusion Framework for Goal-Conditioned Imitation Learning in Robotic Long-Horizon Action Modeling" 

## Installation Guide

First create a conda environment using the following command

```bash
sh install.sh
```

During this process two additional packages will be installed:

- [Relay Policy Learning](https://github.com/google-research/relay-policy-learning)

To add relay_kitchen environment to the PYTHONPATH run the following commands:

```
conda develop <path to your relay-policy-learning directory>
conda develop <path to your relay-policy-learning directory>/adept_envs
conda develop <path to your relay-policy-learning directory>/adept_envs/adept_envs
```

**Dataset**

To download the dataset for the Relay Kitchen and the Block Push environment from the given link and repository, and adjust the data paths in the ```franka_kitchen_main_config.yaml``` and ```block_push_main_config.yaml``` files, follow these steps:

1. Download the dataset: Go to the [link](https://osf.io/q3dx2/) from [play-to-policy](https://github.com/jeffacce/play-to-policy) and download the dataset for the Relay Kitchen and Block Push environments.

2. Unzip the dataset: After downloading, unzip the dataset file and store it.

3. Adjust data paths in the configuration files:

Open the ```./configs/franka_kitchen_main_config.yaml``` and set the data_path argument to ```<path_to_dataset>/relay_kitchen```.
Open the ./configs/block_push_main_config.yaml and set the data_path argument to ```<path_to_dataset>/multimodal_push_fixed_target```
After adjusting the data paths in both configuration files, you should be ready to use the datasets in the respective environments.

---

### Train an agent

There exist the general ```training.py``` file to train a novel agent and evaluate its performance after
the training process. 
A new agent can be trained using the following command:

```bash
[beso]$ conda activate play 
(play)[beso]$ python scripts/training.py 
```
To train the CFG-variant of BESO change the following parameter:
```bash
[beso]$ conda activate play 
(play)[beso]$ python scripts/training.py cond_mask_prob=0.1
```

We can easily train the agent on 10 seeds sequentially by using:
```bash
[beso]$ conda activate play 
(play)[beso]$ python scripts/training.py --multirun seed=1,2,3,4,5,6,7,8,9,10
```

Please note, that we are using wandb to log the training of our model in this repo. Thus one need to adjust, the 
wandb variable in the main config file with your wandb entity and project name. 

---

### Evaluation

We provide several pre-trained models for testing under ```trained_models```.
If you want to evaluate a model and change its inference parameters you can run the following script:
```bash
python scripts/evaluate.py
```
To change parameters for diffusion sampling, check out out ```configs/evaluate_kitchen``` and ```configs/evaluate_blocks```, where there is an detailed overview of all inference parameters. Below we provide an overview of important parameters and the available implementations for each. 


### Acknowledgements

This repo relies on the following existing codebases:

- The goal-conditioned variants of the environments are based on [play-to-policy](https://github.com/jeffacce/play-to-policy).
- The inital environments are adapted from [Relay Policy Learning](https://github.com/google-research/relay-policy-learning), [IBC](https://github.com/google-research/ibc) and [BET](https://github.com/notmahi/bet).
- The continuous time diffusion model is adapted from [k-diffusion](https://github.com/crowsonkb/k-diffusion) together with all sampler implementations. 
- the ```score_gpt``` class is adapted from [miniGPT](https://github.com/karpathy/minGPT).
- A few samplers are have been imported from [dpm-solver](https://github.com/LuChengTHU/dpm-solver)

---
