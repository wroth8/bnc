# On Resource-Efficient Bayesian Network Classifiers and Deep Neural Networks

This directory contains Python code using Tensorflow to reproduce the experiments from our paper

```
@INPROCEEDINGS{Roth2020b,
    title = {On Resource-Efficient Bayesian Network Classifiers and Deep Neural Networks},
    author = {Wolfgang Roth and G{\"{u}}nther Schindler and Holger Fr{\"{o}}ning and Franz Pernkopf},
    booktitle = {International Conference on Pattern Recognition (ICPR)},
    year = {2020 (accepted for publication)},
}
```

This paper is also available at <https://arxiv.org/abs/2010.11773>

## Usage

1. Clone this repository: `git clone --recurse-submodules https://github.com/wroth8/bnc.git`
2. Setup a Python environment (we used Python 3.7) with Tensorflow (we used Tensorflow 2.2). Alternatively, create a virtual environment from the included environment.yml and activate it.
    1. Create using conda: `conda env create -f environment.yml`
    2. Activate using conda: `conda activate icpr-2020-bnc-dnn`
3. Run the desired experiment, for instance, `./run_experiment_bnc_nb.sh`.
Uncomment the desired lines in the corresponding `.sh`-files to run experiments on different datasets and using different settings.
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
Most experiments were conducted on the CPU by setting the environment variable `CUDA_VISIBLE_DEVICES=""`, i.e., no GPU is required.
Only some CNN experiments for larger numbers of channels were conducted using a GPU, although they would also finish on a CPU in a moderate amount of time.

Note that the individual `.sh`-files provide more details about how individual experiments were conducted and to which figures in the paper each individual experiment contributed.

## Experiments
### Quantization for Bayesian networks with naive Bayes structure
Run `./run_experiment_bnc_nb.sh`

### Quantization for Bayesian networks with TAN structure
Run `./run_experiment_bnc_tan.sh`

### Quantization for fully connected DNNs
Run `./run_experiment_fc_dnn.sh`

### Quantization for CNNs (memory matched to Bayesian network classifiers)
Run `./run_experiment_cnn_memory.sh`

### Quantization for CNNs (#operations matched to Bayesian network classifiers)
Run `./run_experiment_cnn_operations.sh`

### Model-size-aware TAN structure learning for Bayesian network classifiers
Run `./run_experiment_tan_structure_learning.sh`

In the paper, we use the `TAN Subset` setup from the PGM-2020 Paper.
Note that the `max_augmenting_features` parameter in the code corresponds to the number of possible parent sets *including* the "no additional parent" option (i.e., class variable only).
Therefore, `max_augmenting_features=9` corresponds to 8 possible *additional* parents as stated in the paper.

We also provide code for the `TAN All` setup which was not used for the ICPR-2020 paper (simply uncomment the corresponding lines).