# Restricted Boltzmann Machines (RBMs) in PyTorch

> **Author:** [Gabriel Bianconi](http://www.gabrielbianconi.com/)

## Overview

This project implements Restricted Boltzmann Machines (RBMs) using PyTorch (see `rbm.py`). Our implementation includes momentum, weight decay, L2 regularization, and CD-*k* contrastive divergence. We also provide support for CPU and GPU (CUDA) calculations.

In addition, we provide an example file applying our model to the MNIST dataset (see `mnist_dataset.py`). The example trains an RBM, uses the trained model to extract features from the images, and finally uses a SciPy-based logistic regression for classification. It achieves 92.8% classification accuracy (this is obviously not a cutting-edge model).
