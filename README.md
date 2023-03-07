<!-- ![workflow badge](https://github.com/MadryLab/trak/blob/main/.github/workflows/python-package.yml/badge.svg) -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/1234.56789) -->
# TRAK: Understanding Model Predictions at Scale

## Installation

### Requirements
- CUDA **toolkit 11 + 7.5 <= GCC <= 10
- CUDA **toolkit** 12 +  GCC >= 7.5

A compatible combination can be installed with `conda`, but refer to the  [FAQ](#faq) for faster/more customizable options.
```bash
conda install -c conda-forge gcc_linux-64==9.3.0 gxx_linux-64=9.3.0 cudatoolkit-dev
export CXX=x86_64-conda-linux-gnu-g++ CC=x86_64-conda-linux-gnu-gcc
```

### Install

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
## FAQ

### How to install `nvcc` (Cuda toolkit) ?

**Version required**: CUDA >= 10.0

**Instructions (Pick one option)**:

- Some machine might already have been setup with the coda toolkit. You can run `nvcc` in a terminal and check if it already exists. If you have a compatible version then you can proceed with the installation
- If you are logged in an unversity/company shared cluster, there is most of the time a way to enable/load a version of cuda tookit without having to install it. On clusters using `modulefile`, the command `module avail` will show you what is available to you. When in doubt, plese refer to the maintainers/documentation of your cluster
- Using conda: `conda install -c conda-forge cudatoolkit-dev`
- If you are `root` on your machine or feel confident with configuring the installation you can follow Nvidia instructions: https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html


### How to install GCC ?

**Version required**:
- CUDA 11: 7.5 <= version <= 10
- CUDA 12: version >= 7.5

**Instructions (Pick one option)**:

- Most Operating System come with `gcc` preinstalled. You can run `gcc --version` in a terminal to check if it's the case on your machine and which version you have. If you have a compatible version then you can proceed with the installation
- Using `conda`
  1. Install `gcc and g++`: `conda install gcc_linux-64==9.3.0 gxx_linux-64=9.3.0`
  2. Enable the compiler before runing `pip install`: `export CXX=x86_64-conda-linux-gnu-g++ CC=x86_64-conda-linux-gnu-gcc`. **This has to be done in the same terminal**
- If your operating ships with an incompatible compiler they usually let you install other version alongside what comes by default. Here is an example for ubuntu and gcc 10:
  1. Add repository: `sudo add-apt-repository ppa:ubuntu-toolchain-r/test`
  2. Update list of packages: `sudo apt update`
  3. Download/install gcc 10: `sudo apt install gcc-10 g++-10`
  4. Enable the compiler before runing `pip install`: `export CXX=g++10 CC=gcc-10`. **This has to be done in the same terminal**
