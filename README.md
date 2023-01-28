# TRAK: Tracing by Rewinding the After Kernel

## Usage

### Setting up TRAK scorer
```python
from trak import TRAKer

traker = TRAKer('/tmp', load_from_existing=False)

model = ...
tr_loader = ...

for inds, batch in loader:
  out = model(batch)
  loss = ...
  traker.make_features(loss, model.parameters(), inds=inds)
  # If the loader is sequential, no need for inds

trak.finalize(out_dir='results/', agg=True, cleanup=False)
```

### Evaluating TRAK scores
```python
from trak import TRAKer

traker = TRAKer('/tmp')
traker.load_base('results/base')

model = ...
val_loader = ...
for batch in val_loader:
  out = model(batch)
  loss = ...
  scores = traker.score(loss, model.parameters())
```
