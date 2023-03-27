.. _quickstart user guide:

Quickstart --- get :code:`TRAK` scores for :code:`CIFAR`
===========================================================

In this tutorial, we'll show you how to use the :code:`TRAK` API to compute data
attribution scores for `ResNet-9 <https://github.com/wbaek/torchskeleton>`_ models trained on
:code:`CIFAR-2`. [1]_ While we use a particular model architecture and dataset, the code in this tutorial can be easily adapted to any classification task.

Overall, this tutorial will show you how to:

#. :ref:`Load model checkpoints`

#. :ref:`Set up the :class:\`.TRAKer\` class`

#. :ref:`Compute :code:\`TRAK\` features for training data`

#. :ref:`Compute :code:\`TRAK\` scores for target examples`

#. :ref:`Visualize the attributions found by TRAK`

#. :ref:`Bonus: Evaluate counterfactuals`


You can also try this tutorial as a
`Jupyter notebook <https://github.com/MadryLab/trak/blob/main/examples/cifar2_correlation.ipynb>`_.
All computations take roughly fifteen minutes in total
on a single A100 GPU.

Let's get started!


.. [1] A subset of the `CIFAR-10 <https://en.wikipedia.org/wiki/CIFAR-10>`_ dataset containing only the "cat" and "dog" classes

Load model checkpoints
----------------------

First, you need models to apply
:code:`TRAK` to. You can either use the script below to train three
ResNet-9 models on :code:`CIFAR-2` and save the checkpoints (e.g., :code:`state_dict()`\ s), or
use your own checkpoint. (In fact, in this tutorial you can replace ResNet-9 + CIFAR-2 with any architecture + classification task of your choice.)

.. raw:: html

   <details>
   <summary><a>Training code for CIFAR-2</a></summary>

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

For the remaining steps, we'll assume you have :code:`N` model
checkpoints in :code:`./checkpoints`:

.. code-block:: python

    import torch
    from pathlib import Path

    ckpt_files = list(Path('./checkpoints').rglob('*.pt'))
    ckpts = [torch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]

Set up the :class:`.TRAKer` class
---------------------------------

The :class:`.TRAKer` class is the entry point to the :code:`TRAK` API. Construct it by calling :code:`__init__()` with three arguments:

* a :code:`model` (a :code:`torch.nn.Module` instance) --- this is the model architecture/class that you want to compute attributions for. Note that this model you pass in does not need to be initialized (we'll do that separately below).

* a :code:`task` (a string or a :class:`.AbstractModelOutput` instance) --- this
  specifies the type of learning task you want to attribue with :code:`TRAK`,
  e.g. image classification, language modeling, CLIP-style contrastive learning, etc.

* a :code:`train_set_size` (an integer) --- the size of the training set you want to keep trak of


.. code-block:: python

    from trak import TRAKer

    # Replace with your choice of model constructor
    model = construct_rn9().to(memory_format=torch.channels_last).cuda().eval()

    traker = TRAKer(model=model,
                    task='image_classification',
                    train_set_size=10_000)  # CIFAR-2 has 10,000 train examples

By default, all metadata and arrays created by :class:`.TRAKer` are stored in
:code:`./trak_results`. You can override this by specifying a custom
:code:`save_dir` to :class:`.TRAKer`.

In addition, you can specify the dimension of the features used by :code:`TRAK` with the :code:`proj_dim`
argument, e.g.,

.. code-block:: python

    traker = TRAKer(..., proj_dim=2048)  # default dimension is 2048

(For the curious, this corresponds to the dimension of the output of random projections in our algorithm.
We recommend :code:`proj_dim` between 1,000 and 40,000.)

For more customizations, check out the :ref:`API reference`.


Compute :code:`TRAK` features for training data
--------------------------------------------

Now that we have constructed a  :class:`.TRAKer` object, let's use it to process the training data. For that, we'll need a data loader:[2]_

.. code-block:: python

    # Replace with your choice of data loader (should be deterministic ordering)
    loader_train = get_dataloader(batch_size=128, split='train')

We process the training examples by calling :meth:`.featurize`:

.. code-block:: python
    :linenos:

    from tqdm import tqdm

    for model_id, ckpt in enumerate(tqdm(ckpts)):
        # TRAKer loads the provided checkpoint and also associates
        # the provided (unique) model_id with the checkpoint.
        traker.load_checkpoint(ckpt, model_id=model_id)

        for batch in loader_train:
            # TRAKer computes features corresponding to the batch of examples,
            # using the checkpoint loaded above.
            traker.featurize(batch=batch, num_samples=batch[0].shape[0])

    # Tells TRAKer that we've given it all the information, at which point
    # TRAKer does some post-processing to get ready for the next step
    # (scoring target examples).
    traker.finalize_features()

.. note::

    Here we assume that the data loader we are using is **not** shuffled,
    so we only need to specify how many samples are in batch.
    Alternatively, we can use
    a shuffled data loader, and pass in :code:`inds` instead of :code:`num_samples`
    to :meth:`.featurize`. In that case, :code:`inds` should be an array of the same
    length as the batch, specifying the indices of the examples in the batch within
    the training dataset.


Above, we sequentially iterate over multiple model checkpoints
.. note::
    While you can still compute :code:`TRAK` with a single checkpoint, using multiple checkpoints significantly improves TRAK's performance. See our

But you can also---and we recommend you to---parallelize this step across multiple jobs.
All you have to do is  initialize a different :class:`.TRAKer` object with the same
:code:`save_dir` within each job and specify the appropriate :code:`model_id` when calling
:meth:`.load_checkpoint`.
For more details, check out how to :ref:`SLURM tutorial`.


.. [2] Again, we use the methods defined in :ref:`Save model checkpoints`.


Compute :code:`TRAK` scores for target examples
---------------------------------------

Finally, we are ready to compute attribution scores.
To do this, you need to choose a set of target examples that you want to attribute.
For the purpose of this tutorial, let's make the
targets be the entire validation set:

.. code-block:: python

    loader_targets = get_dataloader(batch_size=batch_size, split='val')

As before, we iterate over checkpoints and batches of data:

.. code-block:: python
    :linenos:

    for model_id, ckpt in enumerate(tqdm(ckpts)):
        traker.start_scoring_checkpoint(ckpt,
                                        model_id=model_id,
                                        num_targets=len(loader_targets.indices))
        for batch in loader_targets:
            traker.score(batch=batch, num_samples=batch[0].shape[0])

    scores = traker.finalize_scores()

Here, :meth:`.start_scoring_checkpoint` has a similar function to
:meth:`.load_checkpoint` used when featuring the training set; it prepares the
:class:`.TRAKer` by loading the checkpoint and initializing internal data structures.
The :meth:`.score` method is analogous to
:meth:`.featurize`; it processes the target batch and computes
the corresponding features.

.. note::

    Be careful that you provide the **same** :code:`model_id` for each checkpoint as
    in the featurizing step---:code:`TRAK` will **not** check that you did that.
    If you use the wrong :code:`model_id`\ s, :code:`TRAK` will silently fail.

    P.S.: If you know of a clean, robust way to hash model parameters to detect a changed checkpoint,
    open an issue on github and we can add an :code:`assert` to check for :code:`model_id`
    consistency.

The final line above returns :code:`TRAK` scores as a :code:`numpy.array` from the
:meth:`.finalize_scores` method.

That's it!
Once you have your model(s) and your data, just a few API-calls to TRAK
let's you compute data attribution scores.


Visualize the attributions found by TRAK
---------------------------

Let's take a look at what the attribution scores look like.
TODO: add images below code snipeets once we have them; some text to explain what's going on

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


Bonus: Evaluate counterfactuals
-------------------------------

In our paper, we introduce a quantitative way of evaluating data attribution methods using
what we called the *linear datamodeling score*.
Intuitively, this score is a number between 0 and 1 indicating how *counterfactually predictive* the computed attribution scores are.
For example, this is used in our main evaluation (Figure 1 from our paper):

.. image:: assets/main_figure.png

Computing this score requires having an (independent) set of models trained on random subsets of the training dataset. For our CIFAR-2 example, we provide pre-computed data which can be downloaded as in the code below:


.. code-block:: python

    from scipy.stats import spearmanr

    def eval_correlations(scores, tmp_path):
        masks_url = 'https://www.dropbox.com/s/2nmcjaftdavyg0m/mask.npy?dl=1'
        margins_url = 'https://www.dropbox.com/s/tc3r3c3kgna2h27/val_margins.npy?dl=1'

        masks_path = Path(tmp_path).joinpath('mask.npy')
        wget.download(masks_url, out=str(masks_path), bar=None)

        # Boolean matrix of size [num models] x [num training examples]
        # indicating the random subset on which the model was trained on
        masks = torch.as_tensor(np.load(masks_path, mmap_mode='r')).float()

        margins_path = Path(tmp_path).joinpath('val_margins.npy')
        wget.download(margins_url, out=str(margins_path), bar=None)

        # Float matrix of size [num models] x [num targets]
        # indicating the random subset on which the model was trained on
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


For our example, the above code outputs:

.. code-block:: python

    Correlation: 0.0629

(Note: the exact value might flucutate a little bit depending on the particular checkpoints used, but should still concentrate around the above value. Also, the above was based on :code:`TRAK` computed over three checkpoints; the score will increase with more checkpoints used.)