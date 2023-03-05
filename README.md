<!-- ![workflow badge](https://github.com/MadryLab/trak/blob/main/.github/workflows/python-package.yml/badge.svg) -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/1234.56789) -->
# TRAK: Understanding Model Predictions at Scale

## Usage

### Setting up TRAK scorer
```python
from traker import TRAKer

model, checkpoints = ...
train_loader = ...

trak = TRAKer(model=model,
              task='image_classification',
              train_set_size=50_000,
              device='cuda:0')

for model_id, checkpoint in enumerate(checkpoints):
  trak.load_checkpoint(ckeckpoint, model_id=model_id)
  for batch in loader_train:
      trak.featurize(batch=batch, num_samples=loader_train.batch_size)
trak.finalize_features()
```

### Evaluating TRAK scores
```python
from traker import TRAKer

model, checkpoints = ...
val_loader = ...

trak = TRAKer(model=model,
              task='image_classification',
              train_set_size=50_000,
              device='cuda:0')

for model_id, checkpoint in enumerate(checkpoints):
  trak.load_checkpoint(ckeckpoint, model_id=model_id)
  for batch in val_loader:
    trak.score(batch=batch, num_samples=loader_val.batch_size)

scores = trak.finalize_scores()
```
