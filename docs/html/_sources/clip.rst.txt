.. _CLIP tutorial:

Applying :code:`TRAK` to a custom task #3: CLIP
==================================================================================

In this tutorial, we'll show another example of applying :code:`TRAK` to a new
custom task, `CLIP <https://openai.com/research/clip>`_. If you haven't,
you should first check out :ref:`MODELOUTPUT tutorial` to familiarize yourself with the notion of
a model output function and how we implement it inside :code:`TRAK`.


CLIP overview
--------------------------

We'll assume that you're familiar with how CLIP works (having only a rough idea
will be sufficient). For a given image-caption pair :math:`(x, y)`, CLIP outputs an
image embedding :math:`\phi(x)` and a caption embedding :math:`\psi(y)`.

The CLIP training loss tries to align the image embeddings with their corresponding
caption embeddings. In particular, given a batch of :math:`n` examples :math:`\{(x_1,y_1),...,(x_n,y_n)\}`, it computes all :math:`n \times n` pairwise cosine
similarities between the image and text embeddings
:math:`S_{ij}:=\phi(x)\cdot\psi(y)`, and then aims to maximize the :math:`S_{ii}`
terms while minimizing the :math:`S_{ij}` terms for :math:`i\neq j`:

.. math::

    L_\text{CLIP}(x_i, y_i) =
    -\log\left(\frac{\exp(-S_{ii})}{\sum_{j\leq n} \exp(-S_{ij})}\right)
    -\log\left(\frac{\exp(-S_{ii})}{\sum_{j\leq n} \exp(-S_{ji})}\right)


Implementing the model output function
-------------------------------------------------

As in our earlier examples, to apply :code:`TRAK` to this setting, we just need to define
an appropriate model output function.

In our paper, we choose the following model output function:

.. math::

    f_\text{CLIP}(x_i, y_i) =
    -\log\sum_{j\leq n}(\exp(-S_{ii}) - \exp(-S_{ij}))
    -\log\sum_{j\leq n}(\exp(-S_{ii}) - \exp(-S_{ji}))

.. note::
    Intuitively, this choice is motivated by viewing the CLIP loss as a sum of two classification problems (one matching images to their correct captions, and vice versa). Check Section 5.1.1 of our papers for details.

Note that unlike in the classification, this model output evaluated at an example now depends on *other* examples in the batch.
To get the CLIP
embeddings for all the image-caption pairs in the batch, we implement an additional utility method
:meth:`.get_embeddings`. Here, let's just assume we have
access to the arrays :code:`all_img_embeddings` and :code:`all_txt_embeddings`.

Now we are ready to implement :meth:`.CLIPModelOutput.get_output`:

.. code-block:: python

    def get_output(func_model,
                   weights: Iterable[Tensor],
                   buffers: Iterable[Tensor],
                   image: Tensor,
                   label: Tensor):
        image_embeddings, text_embeddings, _ = func_model(weights, buffers,
                                                          image.unsqueeze(0),
                                                          label.unsqueeze(0))

        ii = ch.multinomial(input=ch.arange(N).float(), num_samples=sim_bs, replacement=False)
        result = -ch.logsumexp(-image_embeddings @ (text_embeddings - all_txt_embeddings[ii]).T, dim=1) +\
                 -ch.logsumexp(-text_embeddings @ (image_embeddings - all_img_embeddings[ii]).T, dim=1)
        return result.sum()  # shape of result should be [1], .sum() just removes the extra dimension

Finally, to compute the output-to-loss gradient term, we observe in our paper that we can reduce to the classification case and compute the corresponding probabilities:

.. code-block:: python

    def get_out_to_loss_grad(self, func_model, weights, buffers, batch):
        image_embeddings, text_embeddings, temp = func_model(weights, buffers, *batch)
        if self.temperature is None:
            self.temperature = temp
        res = self.temperature * image_embeddings @ text_embeddings.T
        ps = (self.softmax(res) + self.softmax(res.T)).diag() / 2.
        return (1 - ps).clone().detach()

Note, again, that we are directly implementing the gradient, instead of using
automatic differentiation.


Putting it together
------------------------

Using the above :code:`CLIPModelOutput` implementation, we can compute :code:`TRAK` scores as follows:

.. code-block:: python

    model = ...
    loader_train, loader_val = ...

    traker = TRAKer(model=model,
                    task=CLIPModelOutput, # you can also just pass in "clip"
                    train_set_size=TRAIN_SET_SIZE,
                    save_dir=args.out,
                    device=device,
                    proj_dim=1024)

    traker.load_checkpoint(model.state_dict(), model_id=0)
    for batch in tqdm(loader_train, desc='Featurizing..'):
        batch = [x.cuda() for x in batch]
        traker.featurize(batch=batch, num_samples=batch[0].shape[0])

    traker.finalize_features()

    traker.start_scoring_checkpoint(model.state_dict(), model_id=0, num_targets=VAL_SET_SIZE)
    for batch in tqdm(loader_val, desc='Scoring..'):
        batch = [x.cuda() for x in batch]
        traker.score(batch=batch, num_samples=batch[0].shape[0])

    scores = traker.finalize_scores()


That's all, now you're ready to adapt :code:`TRAK` to your custom tasks!