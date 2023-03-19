<!-- ![workflow badge](https://github.com/MadryLab/trak/blob/main/.github/workflows/python-package.yml/badge.svg) -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/1234.56789) -->
# TRAK: Understanding Model Predictions at Scale

## Usage

### Setting up TRAK scorer
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

### Evaluating TRAK scores
```python
val_loader = ...

for model_id, checkpoint in enumerate(checkpoints):
  traker.start_scoring_checkpoint(ckeckpoint, model_id=model_id, num_targets=...)
  for batch in val_loader:
    traker.score(batch=batch, num_samples=loader_val.batch_size)

scores = traker.finalize_scores()
```

## Installation

To install the version of our package which contains a fast, custom CUDA kernel for the JL projection step, use
```bash
pip install traker[fast]
```
You will need compatible versions of `gcc` and `CUDA toolkit` to install it. See the [installation FAQs](https://trak.csail.mit.edu/html/install.html) for tips regarding this. To install the basic version of our package that requires no compilation, use
```bash
pip install traker
```
