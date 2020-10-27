# Bayesian Network Classifiers

This repository contains the code of research projects concerned with Bayesian network classifiers.

## Differentiable TAN Structure Learning for Bayesian Network Classifiers

The directory `pgm-2020` contains experiments of the paper

```
@INPROCEEDINGS{Roth2020a,
    title = {Differentiable {TAN} Structure Learning for {B}ayesian Network Classifiers},
    author = {Wolfgang Roth and Franz Pernkopf},
    booktitle = {International Conference on Probabilistic Graphical Models (PGM)},
    year = {2020},
}
```

This paper is also available at <https://arxiv.org/abs/2008.09566>.

## On Resource-Efficient Bayesian Network Classifiers and Deep Neural Networks

The directory `icpr-2020` contains experiments of the paper

```
@INPROCEEDINGS{Roth2020b,
    title = {On Resource-Efficient Bayesian Network Classifiers and Deep Neural Networks},
    author = {Wolfgang Roth and G{\"{u}}nther Schindler and Holger Fr{\"{o}}ning and Franz Pernkopf},
    booktitle = {International Conference on Pattern Recognition (ICPR)},
    year = {2020 (accepted for publication)},
}
```

This paper is also available at <https://arxiv.org/abs/2010.11773>.

## Datasets
The directory `datasets` contains several datasets.
These datasets contain discretized features (see `Fayyad and Irani (1993)`).
Files ending with `_mi` contain precomputed mutual information graphs required to compute the Chow-Liu TAN structure (see `Friedman et al. (1997)`).
