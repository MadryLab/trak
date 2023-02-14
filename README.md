<!-- ![workflow badge](https://github.com/MadryLab/trak/blob/main/.github/workflows/python-package.yml/badge.svg) -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/1234.56789) -->
# TRAK: Understanding Model Predictions at Scale

## Usage

### Setting up TRAK scorer

```python
from traker import TRAKer, CrossEntropyModelOutput

model = ...
tr_loader = ...

model_output_fn = CrossEntropyModelOutput()
traker = FunctionalTRAKer('/tmp', model, model_output_fn)

# if using functorch for paralellizing per-sample gradients
func_model, weights, buffers = make_functional_with_buffers(model)


def compute_model_output(weights, buffers, batch):
    batch = images, labels
    logits = func_model(weights, buffers, images)
    return model_output_fn(logits, labels)


# otherwise (vanilla pytorch)
def compute_model_output(model, batch):
    batch = images, labels
    logits = model(images)
    return model_output_fn(logits, labels)


for batch in loader:
    traker.featurize(compute_model_output, weights, buffers, batch)

trak.finalize(out_dir='results/', agg=True, cleanup=False)
```

### Evaluating TRAK scores

```python
from traker import TRAKer

traker = TRAKer('/tmp')
traker.load_base('results/base')

model = ...
val_loader = ...
for batch in val_loader:
    out = model(batch)
    loss = ...
    scores = traker.score(loss, model.parameters())
```
