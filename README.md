[![Python package](https://github.com/MadryLab/trak/actions/workflows/python-package.yml/badge.svg)](https://github.com/MadryLab/trak/actions/workflows/python-package.yml)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/1234.56789) -->
# TRAK: Understanding Model Predictions at Scale
<!-- [[arxiv]](TODO)
[[blog post]](TODO) -->
[[website]](trak.csail.mit.edu)
[[PyPI]](https://pypi.org/project/traker/)
<!-- [[Twitter thread]](TODO) -->

![Main figure](/docs/assets/main_figure.png)
`TRAK`: an effective, efficient data attribution method.

## Abstract

The goal of *data attribution* is to trace model predictions back to the
training data. Prior approaches to this task exhibit a strict tradeoff between
computational demand and efficacy - e.g., methods that are effective for deep
neural networks require training thousands of models, making them impractical
for large models or datasets. In this work, we introduce TRAK (Tracing with the
Randomly-Projected After Kernel), a data attribution method for
overparameterized models that brings us closer to the best of both worlds. Using
only a handful of model checkpoints, TRAK matches the performance of attribution
methods that use thousands of trained models, reducing costs by up to three
orders of magnitude.  We demonstrate the utility of TRAK by applying it to a
variety of large-scale settings: to study CLIP models; to study large language
models (MT5-small); and to accelerate model comparison algorithms.

## Usage

### Setting up the `TRAK` scorer

```python
from trak import TRAKer

model, checkpoints = ...
train_loader = ...

traker = TRAKer(model=model, task='image_classification', train_set_size=...)

for model_id, checkpoint in enumerate(checkpoints):
  traker.load_checkpoint(ckeckpoint, model_id=model_id)
  for batch in loader_train:
      traker.featurize(batch=batch, num_samples=loader_train.batch_size)
traker.finalize_features()
```

### Getting `TRAK` scores

```python
val_loader = ...

for model_id, checkpoint in enumerate(checkpoints):
  traker.start_scoring_checkpoint(ckeckpoint, model_id=model_id, num_targets=...)
  for batch in val_loader:
    traker.score(batch=batch, num_samples=loader_val.batch_size)

scores = traker.finalize_scores()
```

## Installation

To install the version of our package which contains a fast, custom `CUDA`
kernel for the JL projection step, use
```bash
pip install traker[fast]
```
You will need compatible versions of `gcc` and `CUDA toolkit` to install it. See
the [installation FAQs](https://trak.csail.mit.edu/html/install.html) for tips
regarding this. To install the basic version of our package that requires no
compilation, use
```bash
pip install traker
```
