# Differentiable TAN Structure Learning for Bayesian Network Classifiers

This directory contains Python code using Tensorflow to reproduce the experiments from our paper

```
@INPROCEEDINGS{Roth2020,
    title = {Differentiable {TAN} Structure Learning for {B}ayesian Network Classifiers},
    author = {Wolfgang Roth and Franz Pernkopf},
    booktitle = {International Conference on Probabilistic Graphical Models (PGM)},
    year = {2020 (accepted for publication)},
}
```

This paper is also available at `TODO: arxiv link`.

## Usage

1. Clone this repository: `git clone https://github.com/wroth8/bnc.git`
2. Setup a Python environment (we used Python 3.7) with Tensorflow (we used Tensorflow 2.2). Alternatively, create a virtual environment from the included environment.yml and activate it.
    1. Create using conda: `conda env create -f environment.yml`
    2. Activate using conda: `conda activate pgm-2020-structure-learning`
3. Run the desired experiment, for instance, `./run_experiment_naive_bayes.sh`.
Uncomment the desired lines in the corresponding `.sh`-files to run experiments on different datasets.
The experiments are designed to be executed using slurm, i.e., by setting the environment variable `SLURM_ARRAY_TASK_ID`, different random hyperparameter configurations are used.
It should be self explanatory to set hyperparameters to specific values in the Python code, for instance, simply replace

```
bnc_hybrid_tradeoff = random_bnc_hybrid_tradeoff[random_param_idx]
```

by 

```
bnc_hybrid_tradeoff = 123.4
```

Note that experiments write results to a `tensorboard` and a `stats` directory within the experiment directory specified by the parameter `--experiment-dir`.
Make sure that the `stats` directory exists, otherwise the experiment will crash right before finishing when results are written to the `stats` directory.
All experiments were conducted on the CPU by setting the environment variable `CUDA_VISIBLE_DEVICES=""`, i.e., no GPU is required.


## Experiments
### Naive Bayes
Run `run_experiment_naive_bayes.sh`.
To obtain the ML test error, uncomment the lines that initialize the parameters with ML parameters, and observe the initial test error.

### TAN Random
Run `run_experiment_tan_random.sh`.

### TAN Chow-Liu
Run `run_experiment_tan_chow_liu.sh`.
To obtain the ML test error, uncomment the lines that initialize the parameters with ML parameters, and observe the initial test error.

### TAN Subset
Run `run_experiment_tan_subset.sh`.
Note that there is a discrepancy between the notation in the paper and the experiment source code.
In particular, the parameter `max_augmenting_features` in the source code refers to the value `K` in the paper.
However, `max_augmenting_features` is always 1 greater than `K` in the paper, because the no-augmenting-parent option (only the class variable is a parent of the feature variable in the Bayesian network) is also counted as 1 in the experiment code.
Therefore, `max_augmenting_features={3,6,9}` corresponds to `K={2,5,8}` in the paper.

### TAN All
Run `run_experiment_tan_all.sh`.
This experiment was only conducted on `letter` and `satimage`, because the number of features of `usps` and `mnist` is too large.

### TAN Heuristic
Run `run_experiment_tan_heuristic.sh`.
Note that the same discrepancy as in TAN Subset between `max_augmenting_features` and `K` exists.
Moreover, note that we actually implemented four different feature orderings, but in the paper we only reported results for feature orderings {0,2,3} because feature orderings 0 and 1 performed similarly.

### Recover Chow-Liu
Run `run_experiment_recover_chow_liu.sh`.
This experiment maximizes the negative likelihood loss to see whether the Chow-Liu structure can be recovered.
