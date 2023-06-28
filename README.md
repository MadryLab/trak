[![arXiv](https://img.shields.io/badge/arXiv-2303.14186-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2303.14186)
[![PyPI version](https://badge.fury.io/py/traker.svg)](https://badge.fury.io/py/traker)
[![Documentation Status](https://readthedocs.org/projects/trak/badge/?version=latest)](https://trak.readthedocs.io/en/latest/?badge=latest)

# TRAK: Attributing Model Behavior at Scale

[[docs & tutorials]](https://trak.readthedocs.io/en/latest/)
[[blog post]](https://gradientscience.org/trak/)
[[website]](https://trak.csail.mit.edu)

In our [paper](https://arxiv.org/abs/2303.14186), we introduce a new data attribution method called `TRAK` (Tracing with the
Randomly-Projected After Kernel). Using `TRAK`, you can make  accurate
counterfactual predictions (e.g., answers to questions of the form “what would
happen to this prediction if these examples are removed from the training set?").
Computing  data attribution with  TRAK is 2-3 orders of magnitude cheaper than
comparably effective methods, e.g., see our evaluation on:

![Main figure](/docs/assets/main_figure.png)

## Usage

- [quickstart (tutorial & notebook)](https://trak.readthedocs.io/en/latest/quickstart.html)
- [pre-computed TRAK scores for CIFAR-10 (Google Colab notebook)](https://colab.research.google.com/drive/1Mlpzno97qpI3UC1jpOATXEHPD-lzn9Wg?usp=sharing)

Check [our docs](https://trak.readthedocs.io/en/latest/) for more detailed examples and
tutorials on how to use `TRAK`.  Below, we provide a brief blueprint of using `TRAK`'s API to compute attribution scores.

### Make a `TRAKer` instance

```python
from trak import TRAKer

model, checkpoints = ...
train_loader = ...

traker = TRAKer(model=model, task='image_classification', train_set_size=...)
```

### Compute `TRAK` features on training data

```python
for model_id, checkpoint in enumerate(checkpoints):
  traker.load_checkpoint(checkpoint, model_id=model_id)
  for batch in loader_train:
      # batch should be a tuple of inputs and labels
      traker.featurize(batch=batch, ...)
traker.finalize_features()
```

### Compute `TRAK` scores for target examples

```python
targets_loader = ...

for model_id, checkpoint in enumerate(checkpoints):
  traker.start_scoring_checkpoint(checkpoint,
                                  model_id=model_id,
                                  exp_name='test',
                                  num_targets=...)
  for batch in targets_loader:
    traker.score(batch=batch, ...)

scores = traker.finalize_scores(exp_name='test')
```
Check out the [quickstart](https://trak.readthedocs.io/en/latest/quickstart.html) for a complete ready-to-run example notebook.


## Examples
You can find several end-to-end examples in the `examples/` directory.

## Citation
If you use this code in your work, please cite using the following BibTeX entry:
```
@inproceedings{park2023trak,
  title = {TRAK: Attributing Model Behavior at Scale},
  author = {Sung Min Park and Kristian Georgiev and Andrew Ilyas and Guillaume Leclerc and Aleksander Madry},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2023}
}
```

## Installation

To install the version of our package which contains a fast, custom `CUDA`
kernel for the JL projection step, use
```bash
pip install traker[fast]
```
You will need compatible versions of `gcc` and `CUDA toolkit` to install it. See
the [installation FAQs](https://trak.readthedocs.io/en/latest/install.html) for tips
regarding this. To install the basic version of our package that requires no
compilation, use
```bash
pip install traker
```

## Questions?

Please send an email to trak@mit.edu

## Maintainers

[Kristian Georgiev](https://twitter.com/kris_georgiev1)<br>
[Andrew Ilyas](https://twitter.com/andrew_ilyas)<br>
[Sung Min Park](https://twitter.com/smsampark)
