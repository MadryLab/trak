<!-- ![workflow badge](https://github.com/MadryLab/trak/blob/main/.github/workflows/python-package.yml/badge.svg) -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/1234.56789) -->
# TRAK: Understanding Model Predictions at Scale

## Usage

### Setting up TRAK scorer
```python
from traker import TRAKer, CrossEntropyModelOutput

model, checkpoints = ...
train_loader = ...

model_output_fn = CrossEntropyModelOutput()
trak = TRAKer(model, save_dir='/tmp', device='cuda:0')

func_model, weights, buffers = make_functional_with_buffers(model)
def compute_model_output(weights, buffers, image, label):
  out = func_model(weights, buffers, image.unsqueeze(0))
  return modelout_fn.get_output(out, label.unsqueeze(0)).sum()

def compute_out_to_loss(weights, buffers, images, labels):
    out = func_model(weights, buffers, images)
    return modelout_fn.get_out_to_loss(out, labels)

for model_id, checkpoint in enumerate(checkpoints):
  # load checkpoint here, get new weights & buffers ...
  trak.load_params(model_params=(weights, buffers))
  for batch in loader_train:
      inds = ...  # if loading in sequential order, this can be skipped
      trak.featurize(out_fn=compute_outputs, loss_fn=compute_out_to_loss,
                     batch=batch, model_id=model_id, inds=inds)
trak.finalize()
trak.save()
```

### Evaluating TRAK scores
```python
from traker import TRAKer

model, checkpoints = ...
val_loader = ...

trak = TRAKer('/tmp')
trak.load()

for model_id, checkpoint in enumerate(checkpoints):
  # load checkpoint here ...
  for batch in val_loader:
    scores.append(trak.score(out_fn=compute_outputs, batch=batch,
                             model=model, model_id=model_id).cpu())
```
