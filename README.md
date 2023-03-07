<!-- ![workflow badge](https://github.com/MadryLab/trak/blob/main/.github/workflows/python-package.yml/badge.svg) -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/1234.56789) -->
# TRAK: Understanding Model Predictions at Scale

## Installation

### 0. TLDR

`conda install -c conda-forge cudatoolkit-dev; pip install traker`

If you are not using conda or if this fails follow the next steps

### 1. Install pytorch

Use the method of your choice to install Pytorch. See http://pytorch.org for suggestions.

### 2. Obtain the development version of Nvidia CUDA

- If `nvcc` is available in a terminal you are already done and can skip this step
- If you are using `conda` this can be done using `conda install -c conda-forge cudatoolkit-dev`
- Alternatively you can follow official instructions from Nvidia for your operating system: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

### 3. Check you have a compatible compiler

There is a bug in `CUDA 11` when used in combination with `GCC 11.3+`. If you are in this scenario you can either:
- Upgrade to CUDA `12.0+`
- Install and enable `gcc-10`. On ubuntu this can be done this way:
```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-10 g++-10
export CC=gcc-10 CXX=g++-10
```

### 4. Install and build `traker`

```bash
pip install traker
```

## Usage

### Setting up TRAK scorer
```python
from trak import TRAKer

model, checkpoints = ...
train_loader = ...

traker = TRAKer(model=model,
              task='image_classification',
              train_set_size=50_000,
              device='cuda:0')

for model_id, checkpoint in enumerate(checkpoints):
  traker.load_checkpoint(ckeckpoint, model_id=model_id)
  for batch in loader_train:
      traker.featurize(batch=batch, num_samples=loader_train.batch_size)
traker.finalize_features()
```

### Evaluating TRAK scores
```python
from trak import TRAKer

model, checkpoints = ...
val_loader = ...

traker = TRAKer(model=model,
              task='image_classification',
              train_set_size=50_000,
              device='cuda:0')

for model_id, checkpoint in enumerate(checkpoints):
  traker.load_checkpoint(ckeckpoint, model_id=model_id)
  for batch in val_loader:
    traker.score(batch=batch, num_samples=loader_val.batch_size)

scores = traker.finalize_scores()
```
