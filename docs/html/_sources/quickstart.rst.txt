.. _quickstart user guide:

Quickstart --- get :code:`TRAK` scores for :code:`CIFAR`
===========================================================

In this tutorial, we'll show you how to use the :code:`TRAK` API to compute data
attribution scores for `ResNet-9 <https://github.com/wbaek/torchskeleton>`_ on
:code:`CIFAR-2` [1]_. All computations take roughly fifteen minutes in a total
on a single A100 GPU.
Overall, this tutorial will show you how to:

* :ref:`Save model checkpoints`

* :ref:`Set up the :class:\`.TRAKer\` class`

* :ref:`Compute :code:\`TRAK\` features for train data`

* :ref:`Compute :code:\`TRAK\` scores for targets`

* :ref:`Visualize the attributions!`

* :ref:`Extra: evaluate counterfactuals`

Here we'll provide tips and tricks for you to quickly get :code:`TRAK` up and
running; for more details, check the :ref:`API reference`.

.. [1] A subset of the `CIFAR-10 <https://en.wikipedia.org/wiki/CIFAR-10>`_ dataset containing only the "cat" and "dog" classes

Let's get started!

Save model checkpoints
----------------------

First, we need to have model checkpoints (e.g. :code:`state_dict()`\ s) to apply
:code:`TRAK` on. You can either use the script below to train twenty copies of
Resnet-9 on :code:`CIFAR-2` and save the checkpoints, or bring your own
checkpoint (any architecture/dataset combination within image classification
will be fine for the following steps).

.. raw:: html

   <details>
   <summary><a>Check the exact training script we used.</a></summary>

.. code-block:: python

    from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.pipeline.operation import Operation
    from ffcv.transforms import RandomHorizontalFlip, Cutout, \
        RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
    from ffcv.transforms.common import Squeeze

    import os
    import wget
    from tqdm import tqdm
    import numpy as np
    import torch
    from torch.cuda.amp import GradScaler, autocast
    from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
    from torch.optim import SGD, lr_scheduler
    import torchvision

    BETONS = {
            'train': "https://www.dropbox.com/s/zn7jsp2rl09e0fh/train.beton?dl=1",
            'val': "https://www.dropbox.com/s/4p73milxxafv4cm/val.beton?dl=1",
    }

    STATS = {
            'mean': [125.307, 122.961, 113.8575],
            'std': [51.5865, 50.847, 51.255]
    }

    def get_dataloader(batch_size=256,
                    num_workers=8,
                    split='train',  # split \in [train, val]
                    aug_seed=0,
                    should_augment=True,
                    indices=None):
            label_pipeline: List[Operation] = [IntDecoder(),
                                            ToTensor(),
                                            ToDevice(torch.device('cuda:0')),
                                            Squeeze()]
            image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

            if should_augment:
                    image_pipeline.extend([
                            RandomHorizontalFlip(),
                            RandomTranslate(padding=2, fill=tuple(map(int, STATS['mean']))),
                            Cutout(4, tuple(map(int, STATS['std']))),
                    ])

            image_pipeline.extend([
                ToTensor(),
                ToDevice(torch.device('cuda:0'), non_blocking=True),
                ToTorchImage(),
                Convert(torch.float32),
                torchvision.transforms.Normalize(STATS['mean'], STATS['std']),
            ])

            beton_url = BETONS[split]
            beton_path = f'./{split}.beton'
            wget.download(beton_url, out=str(beton_path), bar=None)

            return Loader(beton_path,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        seed=aug_seed,
                        indices=indices,
                        pipelines={'image': image_pipeline, 'label': label_pipeline})


    # Resnet9
    class Mul(torch.nn.Module):
        def __init__(self, weight):
            super(Mul, self).__init__()
            self.weight = weight
        def forward(self, x): return x * self.weight


    class Flatten(torch.nn.Module):
        def forward(self, x): return x.view(x.size(0), -1)


    class Residual(torch.nn.Module):
        def __init__(self, module):
            super(Residual, self).__init__()
            self.module = module
        def forward(self, x): return x + self.module(x)


    def construct_rn9(num_classes=2):
        def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
            return torch.nn.Sequential(
                    torch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                                stride=stride, padding=padding, groups=groups, bias=False),
                    torch.nn.BatchNorm2d(channels_out),
                    torch.nn.ReLU(inplace=True)
            )
        model = torch.nn.Sequential(
            conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
            conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
            Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
            conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2),
            Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
            conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
            torch.nn.AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            torch.nn.Linear(128, num_classes, bias=False),
            Mul(0.2)
        )
        return model

    def train(model, loader, lr=0.4, epochs=100, momentum=0.9, weight_decay=5e-4, lr_peak_epoch=5, label_smoothing=0.0):
        opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        iters_per_epoch = len(loader)
        # Cyclic LR with single triangle
        lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                                [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                                [0, 1, 0])
        scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
        scaler = GradScaler()
        loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

        for ep in range(epochs):
            model_count = 0
            for it, (ims, labs) in enumerate(loader):
                opt.zero_grad(set_to_none=True)
                with autocast():
                    out = model(ims)
                    loss = loss_fn(out, labs)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                scheduler.step()

    os.makedirs('./checkpoints', exist_ok=True)

    for i in tqdm(range(20), desc='Training models..'):
        model = construct_rn9().to(memory_format=torch.channels_last).cuda()
        loader_train = get_dataloader(batch_size=512, split='train')
        train(model, loader_train)
        
        torch.save(model.state_dict(), f'./checkpoints/sd_{i}.pt')

.. raw:: html

   </details>

For the remaining steps, we're going to assume you have :code:`N` model
checkpoints in :code:`./checkpoints`:

.. code-block:: python

    import torch
    from pathlib import Path

    ckpt_files = list(Path('./checkpoints').rglob('*.pt'))
    ckpts = [torch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]

Set up the :class:`.TRAKer` class
---------------------------------

First, let's initlaize our model. Using the methods from the
section above:

.. code-block:: python

    model = construct_rn9().to(memory_format=torch.channels_last).cuda().eval()

The :class:`.TRAKer` class is the entry point to the :code:`TRAK` API. The two
most important arguments for its :code:`__init__()` are:

* a model architecture (a :code:`torch.nn.Module` instance)

* a :code:`task` (a string or a :class:`.AbstractModelOutput` instance) --- this
  specifies what type of learning task we are attributing with :code:`TRAK`,
  e.g. image classification, language modeling, CLIP-style contrastive learning
  natural language supervision, etc.



.. code-block:: python

    from trak import TRAKer

    traker = TRAKer(model=model,
                    task='image_classification',
                    train_set_size=10_000)  # CIFAR-2 has 10,000 train examples

We need to specify the size of the train set so that :class:`.TRAKer` can
initialize `memory-maps
<https://numpy.org/doc/stable/reference/generated/numpy.memmap.html>`_ of the
correct size for gradient features and attribution scores.
By default, all metadata and arrays created by :class:`.TRAKer` are stored in
:code:`./trak_results`. You can override this by specifying a custom
:code:`save_dir` to :class:`.TRAKer`.

You can initalize multiple :class:`.TRAKer` objects with the same
:code:`save_dir`. In fact, running multiple :class:`.TRAKer`\ s in parallel, all
writing to the same :code:`save_dir`, is not only supported, but encouraged!
Check out how to :ref:`SLURM tutorial` for more details.

In addition, we can specify the size of the "out" dimension of the
dimensionality reduction step in :code:`TRAK` with the :code:`proj_dim`
argument, e.g.:

.. code-block:: python

    traker = TRAKer(..., proj_dim=4096)

See our paper for ablations on the size of the "out" dimension (maybe
surprisingly, bigger :code:`proj_dim` is not always better!)

Compute :code:`TRAK` features for train data
--------------------------------------------

Now let's process the train data. For that, we'll need a data loader:[2]_

.. code-block:: python

    batch_size = 128
    loader_train = get_dataloader(batch_size=batch_size, split='train')

We're ready to :meth:`.featurize` the training samples:

.. code-block:: python
    :linenos:

    from tqdm import tqdm  # for progress tracking

    for model_id, ckpt in enumerate(tqdm(ckpts)):
        traker.load_checkpoint(ckpt, model_id=model_id)

        for batch in loader_train:
            traker.featurize(batch=batch, num_samples=batch[0].shape[0])

    traker.finalize_features()

Let's analyze this step by step. On line 3, we're iterating over the
checkpoints, assigning a :code:`model_id` (just the checkpoint's index in the
:code:`ckpt` array in this case) for each one. Then, on line 4, the
:meth:`.load_checkpoint` method registers the checkpoint in the :class:`.TRAKer`
class and ties it to the given :code:`model_id`. [3]_ In lines 6 and 7, we are
iterating over the train data, getting gradient features for all examples.
This step involves computing per-example gradients. Finally, in line 9, we
perform some post-processing of the computed features (in particular, we compute
the reweighting term, check our paper if you're curious what that is).

Note that the loader we are using is **not** shuffled. Because of that, we only
need to specify how many samples are in batch, and :class:`TRAKer` writes
sequentially in the memory-map for gradient features. Alternatively, we can use
a shuffled data loader, and pass in :code:`inds` instead of :code:`num_samples`
to :meth:`.featurize`. In that case, :code:`inds` should be an array of the same
length as the batch, specifying the indices of the examples in the batch within
the train data.

.. [2] Again, we use the methods defined in :ref:`Save model checkpoints`. Adapt this as you wish.
.. [3] :code:`model_id`\ s will be important later when we compute scores and need to match gradient features of the train data and targets across checkpoints

Compute :code:`TRAK` scores for targets
---------------------------------------

Finally, time for scoring! For the purpose of this tutorial, let's make the
targets be the entire validation set:

.. code-block:: python

    loader_targets = get_dataloader(batch_size=batch_size, split='val')

Using a similar interface to the featurizing step:

.. code-block:: python
    :linenos:

    for model_id, ckpt in enumerate(tqdm(ckpts)):
        traker.start_scoring_checkpoint(ckpt,
                                        model_id=model_id,
                                        num_targets=len(loader_targets.indices))
        for batch in loader_targets:
            traker.score(batch=batch, num_samples=batch[0].shape[0])

    scores = traker.finalize_scores()

TODO

.. note::

    Be careful that you provide the same :code:`model_id` for each checkpoint as
    in the featurizing step - :code:`TRAK` will **not** check that you did that.
    If you shuffle the :code:`model_id`\ s, you'll not receive an error, but
    you'll get bad results.

    P.S.: if you know a clean, robust way to hash model parameters, open an
    issue on github and we might add an :code:`assert` about :code:`model_id`
    consistency.

Visualize the attributions!
---------------------------

TODO: add images once we have them

.. code-block:: python

    from matplotlib import pyplot as plt

    targets = [1, 2]  # let's look at two validation images
    loader_targets = get_dataloader(batch_size=2, split='val', indices=targets, should_augment=False)

    for batch in loader_targets:
        ims, _ = batch
        ims = (ims - ims.min()) / (ims.max() - ims.min())
        for image in ims:
            plt.figure(figsize=(1.5,1.5))
            plt.imshow(image.cpu().permute([1, 2, 0]).numpy())
            plt.axis('off'); plt.show()

.. code-block:: python

    for target in targets:
        print(f'Top scorers for target {target}')
        loader_top_scorer = get_dataloader(batch_size=3, split='train', indices=scores[target].argsort()[-3:].cpu().numpy(), should_augment=False)
        for batch in loader_top_scorer:
            ims, _ = batch
            ims = (ims - ims.min()) / (ims.max() - ims.min())
            for image in ims:
                plt.figure(figsize=(1.5, 1.5))
                plt.imshow(image.cpu().permute([1, 2, 0]).numpy()); plt.axis('off'); plt.show()


Extra: evaluate counterfactuals
-------------------------------

TODO

Now we can perform an evaluation similar to the one we did to produce Figure 1 in our paper:

.. image:: assets/main_figure.png

.. code-block:: python

    masks = ...
    margins = ...


.. code-block:: python

    from scipy.stats import spearmanr

    def eval_correlations(scores, tmp_path):
        masks_url = 'https://www.dropbox.com/s/2nmcjaftdavyg0m/mask.npy?dl=1'
        margins_url = 'https://www.dropbox.com/s/tc3r3c3kgna2h27/val_margins.npy?dl=1'

        masks_path = Path(tmp_path).joinpath('mask.npy')
        wget.download(masks_url, out=str(masks_path), bar=None)
        # num masks, num train samples
        masks = torch.as_tensor(np.load(masks_path, mmap_mode='r')).float()

        margins_path = Path(tmp_path).joinpath('val_margins.npy')
        wget.download(margins_url, out=str(margins_path), bar=None)
        # num , num val samples
        margins = torch.as_tensor(np.load(margins_path, mmap_mode='r'))

        val_inds = np.arange(2000)
        preds = masks @ scores
        rs = []
        ps = []
        for ind, j in tqdm(enumerate(val_inds)):
            r, p = spearmanr(preds[:, ind], margins[:, j])
            rs.append(r)
            ps.append(p)
        rs, ps = np.array(rs), np.array(ps)
        print(f'Correlation: {rs.mean()} (avg p value {ps.mean()})')
    return rs.mean()

    eval_correlations(scores.cpu(), '.')