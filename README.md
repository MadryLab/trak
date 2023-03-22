[![Python package](https://github.com/MadryLab/trak/actions/workflows/python-package.yml/badge.svg)](https://github.com/MadryLab/trak/actions/workflows/python-package.yml)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/1234.56789) -->

# TRAK: Attributing Model Behavior at Scale

`TRAK` is an effective, efficient data attribution method. Check our
[paper](TODO) and [blog post](TODO) for more details.

[[docs]](https://trak.csail.mit.edu/html/index.html)
[[website]](trak.csail.mit.edu)
[[PyPI]](https://pypi.org/project/traker/)
<!--[[blog post]](TODO) -->
<!--[[Twitter thread]](TODO) -->

![Main figure](/docs/assets/main_figure.png)

## Usage

Check [our docs](https://trak.csail.mit.edu/html) for more detailed examples and
tutorials on how to use `TRAK`.  Below, we provide a brief blueprint of the
steps needed to get `TRAK` attribution scores with our API.

### Setting up the `TRAK` scorer

```python
from trak import TRAKer

model, checkpoints = ...
train_loader = ...

traker = TRAKer(model=model, task='image_classification', train_set_size=...)

for model_id, checkpoint in enumerate(checkpoints):
  traker.load_checkpoint(ckeckpoint, model_id=model_id)
  for batch in loader_train:
      traker.featurize(batch=batch, ...)
traker.finalize_features()
```

### Getting `TRAK` scores

```python
targets_loader = ...

for model_id, checkpoint in enumerate(checkpoints):
  traker.start_scoring_checkpoint(ckeckpoint, model_id=model_id, num_targets=...)
  for batch in targets_loader:
    traker.score(batch=batch, ...)

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
