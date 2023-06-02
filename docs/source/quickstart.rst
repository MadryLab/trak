.. _quickstart user guide:

Quickstart --- get :code:`TRAK` scores for :code:`CIFAR`
===========================================================

.. note::

    Follow along in this `Jupyter notebook
    <https://github.com/MadryLab/trak/blob/main/examples/cifar_quickstart.ipynb>`_.
    If you want to browse pre-computed TRAK scores instead, check out this
    `Colab notebook
    <https://colab.research.google.com/drive/1Mlpzno97qpI3UC1jpOATXEHPD-lzn9Wg?usp=sharing>`_.

In this tutorial, we'll show you how to use the :code:`TRAK` API to compute data
attribution scores for `ResNet-9 <https://github.com/wbaek/torchskeleton>`_ models trained on
:code:`CIFAR-10`. While we use a particular model architecture and dataset, the code in this tutorial can be easily adapted to any classification task.

Overall, this tutorial will show you how to:

#. :ref:`Load model checkpoints`

#. :ref:`Set up the :class:\`.TRAKer\` class`

#. :ref:`Compute :code:\`TRAK\` features for training data`

#. :ref:`Compute :code:\`TRAK\` scores for target examples`


Let's get started!

Load model checkpoints
----------------------

First, you need models to apply :code:`TRAK` to. You can either use the script
below to train three ResNet-9 models on :code:`CIFAR-10` and save the checkpoints
(e.g., :code:`state_dict()`\ s), or use your own checkpoint [1]_. (In fact, in this
tutorial you can replace ResNet-9 + CIFAR-10 with any architecture +
classification task of your choice.)

.. raw:: html

   <details>
   <summary><a>Training code for CIFAR-10</a></summary>

.. code-block:: python

    import os
    from pathlib import Path
    import wget
    from tqdm import tqdm
    import numpy as np
    import torch
    from torch.cuda.amp import GradScaler, autocast
    from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
    from torch.optim import SGD, lr_scheduler
    import torchvision

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


    def construct_rn9(num_classes=10):
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

    def get_dataloader(batch_size=256, num_workers=8, split='train', shuffle=False, augment=True):
        if augment:
            transforms = torchvision.transforms.Compose(
                            [torchvision.transforms.RandomHorizontalFlip(),
                             torchvision.transforms.RandomAffine(0),
                             torchvision.transforms.ToTensor(),
                             torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                              (0.2023, 0.1994, 0.201))])
        else:
            transforms = torchvision.transforms.Compose([
                             torchvision.transforms.ToTensor(),
                             torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                              (0.2023, 0.1994, 0.201))])

        is_train = (split == 'train')
        dataset = torchvision.datasets.CIFAR10(root='/tmp/cifar/',
                                               download=True,
                                               train=is_train,
                                               transform=transforms)

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             shuffle=shuffle,
                                             batch_size=batch_size,
                                             num_workers=num_workers)

        return loader

    def train(model, loader, lr=0.4, epochs=24, momentum=0.9,
              weight_decay=5e-4, lr_peak_epoch=5, label_smoothing=0.0, model_id=0):

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
            for it, (ims, labs) in enumerate(loader):
                ims = ims.cuda()
                labs = labs.cuda()
                opt.zero_grad(set_to_none=True)
                with autocast():
                    out = model(ims)
                    loss = loss_fn(out, labs)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                scheduler.step()
            if ep in [12, 15, 18, 21, 23]:
                torch.save(model.state_dict(), f'./checkpoints/sd_{model_id}_epoch_{ep}.pt')

        return model

    os.makedirs('./checkpoints', exist_ok=True)
    loader_for_training = get_dataloader(batch_size=512, split='train', shuffle=True)

    # you can modify the for loop below to train more models
    for i in tqdm(range(1), desc='Training models..'):
        model = construct_rn9().to(memory_format=torch.channels_last).cuda()
        model = train(model, loader_for_training, model_id=i)


.. raw:: html

   </details>

For the remaining steps, we'll assume you have :code:`N` model
checkpoints in :code:`./checkpoints`:

.. code-block:: python

    import torch
    from pathlib import Path

    ckpt_files = list(Path('./checkpoints').rglob('*.pt'))
    ckpts = [torch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]

.. [1] For our own experiments, we used `ffcv <https://ffcv.io/>`_ to train models faster. Check `this <https://github.com/MadryLab/trak/blob/main/examples/imagenet.py>`_ training script that trains the same ResNet-9 models using :code:`ffcv` dataloaders.

Set up the :class:`.TRAKer` class
---------------------------------

The :class:`.TRAKer` class is the entry point to the :code:`TRAK` API. Construct it by calling :code:`__init__()` with three arguments:

* a :code:`model` (a :code:`torch.nn.Module` instance) --- this is the model architecture/class that you want to compute attributions for. Note that this model you pass in does not need to be initialized (we'll do that separately below).

* a :code:`task` (a string or a :class:`.AbstractModelOutput` instance) --- this
  specifies the type of learning task you want to attribue with :code:`TRAK`,
  e.g. image classification, language modeling, CLIP-style contrastive learning, etc.
  Internally, the task tells :class:`.TRAKer` how to evaluate a given batch of data.

* a :code:`train_set_size` (an integer) --- the size of the training set you want to keep trak of

Let's set up our model and dataset:

.. code-block:: python

    # Replace with your choice of model constructor
    model = construct_rn9().to(memory_format=torch.channels_last).cuda().eval()

    # Replace with your choice of data loader (should be deterministic ordering)
    loader_train = get_dataloader(batch_size=128, split='train')

Now we are ready to start TRAKing our model on the dataset of choice. Let's
initialize the TRAKer object.

.. code-block:: python

    from trak import TRAKer

    traker = TRAKer(model=model,
                    task='image_classification',
                    train_set_size=len(loader_train.dataset))

By default, all metadata and arrays created by :class:`.TRAKer` are stored in
:code:`./trak_results`. You can override this by specifying a custom
:code:`save_dir` to :class:`.TRAKer`.

In addition, you can specify the dimension of the features used by :code:`TRAK` with the :code:`proj_dim`
argument, e.g.,

.. code-block:: python

    traker = TRAKer(..., proj_dim=4096)  # default dimension is 2048

(For the curious, this corresponds to the dimension of the output of random
projections in our algorithm.  We recommend :code:`proj_dim` between 1,000 and
40,000.)

For more customizations, check out the :ref:`API reference`.


Compute :code:`TRAK` features for training data
-----------------------------------------------

Now that we have constructed a  :class:`.TRAKer` object, let's use it to process
the training data. We process the training examples by calling
:meth:`.featurize`:

.. code-block:: python
    :linenos:

    for model_id, ckpt in enumerate(tqdm(ckpts)):
        # TRAKer loads the provided checkpoint and also associates
        # the provided (unique) model_id with the checkpoint.
        traker.load_checkpoint(ckpt, model_id=model_id)

        for batch in loader_train:
            batch = [x.cuda() for x in batch]
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


Compute :code:`TRAK` scores for target examples
-----------------------------------------------

Finally, we are ready to compute attribution scores.
To do this, you need to choose a set of target examples that you want to attribute.
For the purpose of this tutorial, let's make the
targets be the entire validation set:

.. code-block:: python

    loader_targets = get_dataloader(batch_size=batch_size, split='val', augment=False)


As before, we iterate over checkpoints and batches of data:

.. code-block:: python
    :linenos:

    for model_id, ckpt in enumerate(tqdm(ckpts)):
        traker.start_scoring_checkpoint(exp_name='quickstart',
                                        checkpoint=ckpt,
                                        model_id=model_id,
                                        num_targets=len(loader_targets.dataset))
        for batch in loader_targets:
            batch = [x.cuda() for x in batch]
            traker.score(batch=batch, num_samples=batch[0].shape[0])

    scores = traker.finalize_scores(exp_name='quickstart')

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

The final line above returns :code:`TRAK` scores as a :code:`numpy.array` from
the :meth:`.finalize_scores` method. Additionally, :meth:`.finalize_scores`
saves the scores to disk in memory-mapped file (:code:`.mmap` format).

We can visualize some of the top scoring :code:`TRAK` images from the
:code:`scores` array we just computed:

.. image:: assets/trak_scores_quickstart.png
   :alt: Top scoring TRAK images


That's it!
Once you have your model(s) and your data, just a few API-calls to :code:`TRAK``
let's you compute data attribution scores.
