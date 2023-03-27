.. _CLIP model output:

Add  a :code:`task` to :code:`TRAKer` (subclassing :code:`ModelOutput`\ ) --- CLIP
==================================================================================

In this tutorial, we'll go through the process of applying :code:`TRAK` to a new
custom task, using CLIP as an example. Check out :class:`.CLIPModelOutput` for
the final result.

Model output functions
----------------------

Computing :code:`TRAK` attribution scores requires defining an appropriate ``model output
function`` to guide the scoring process. Intuitively, you can just think of it as a
loss function. We derive and discuss model output
functions for multiple tasks (binary and multiclass classification, CLIP loss,
and various NLP tasks, etc) in detail in `our paper <link:TODO>`_.

In short, given the following:

* a training set of examples :math:`S = \{z_1, z_2, \ldots, z_n\}`
  where each :math:`z_i  = (x_i, y_i)` is an input-label pair,
* an example of interest :math:`z`,
* and model parameters :math:`\theta`,

we (implicitly) represent machine learning models with *some* model output
function :math:`f(z;\theta)` --- a function which maps an example of interest
and model parameters to a real number.

.. note::

    One example model output function is the *loss* (e.g. cross-entropy loss)
    that the model incurs on :math:`z`.

The modelout-to-loss gradient (the :math:`Q` term)
--------------------------------------------------

Additionally, the final :code:`TRAK` estimator (see Algorithm 1 in our paper)
uses a ":math:`Q`" term -- an :math:`n\times n` diagonal matrix of “one minus
correct-class probability” terms. In classification, this form of :math:`Q`
comes from the fact that, for :math:`f(z;\theta) = \log\left(\frac{p(z;\theta)}{1 - p(z;\theta)}\right)`,
we have that the gradient of the (cross-entropy) loss :math:`\ell` wrt :math:`f`
is:

.. math::

    \frac{\partial \ell(z;\theta)}{\partial f} = \frac{\partial}{\partial f}
    \log(1 + \exp(-f)) = -\frac{\exp(-f)}{1 + \exp(-f)}  = -(1 - p(z;\theta))

If you are adapting :code:`TRAK` a more exotic task, unrelated to
classification, you might need to modify this term accordingly (for all of our
experiments, this was not necessary).


How we implement model output functions in :code:`TRAK`
-------------------------------------------------------

We provide a dedicated class -- :class:`.AbstractModelOutput` that takes care of
translating models (:code:`torch.Module` instances) into model output functions.
This is achieved via two methods:

* :meth:`.AbstractModelOutput.get_output`
* :meth:`.AbstractModelOutput.get_out_to_loss_grad`

The :meth:`.AbstractModelOutput.get_output` method implements the model output
function. In particular, given a batch of examples :math:`b`, it returns a
vector containing the model outputs for each example in the batch. This is the
function that we pass through PyTorch's :code:`autograd`.

The :meth:`.AbstractModelOutput.get_out_to_loss_grad` implements the *gradient*
of the modelout-to-loss term :math:`Q`. Since for all applications (so far!) we
could analytically derive the gradient of the :math:`Q` term, we "hardcoded"
this in the :code:`get_out_to_loss_grad` method, thus avoiding an additional
gradient computation.

.. note::

    If you find yourself in the (we believe unlikely) situation where you can't
    analytically derive the gradient for :math:`Q`, adapt :meth:`.AbstractModelOutput.get_out_to_loss_grad`
    to follow the structure of :meth:`.AbstractModelOutput.get_output` and pass
    it to :code:`autograd` as well.

What you need to do for a new task
----------------------------------

When we adapt :code:`TRAK` for a new task, we only need to create a new subclass of
:class:`.AbstractModelOutput`. Let's do this by example.

We'll assume that you're familiar with how CLIP works (having only a rough idea
will be sufficient). For a given image-caption pair :math:`(x, y)`, we'll denote
image embeddings as :math:`\phi(x)` and caption embeddings as :math:`\psi(y)`.
The CLIP training loss computes all all :math:`n \times n` pairwise cosine
similarities between the image and text embeddings
:math:`S_{ij}:=\phi(x)\cdot\psi(y)`; it then aims to maximize :math:`S_{ii}`
terms, and minimize :math:`S_{ij}` terms for :math:`i\neq j`:

.. math::

    \ell_{CLIP}(x_i, y_i) =
    -\log\left(\frac{\exp(-S_{ii})}{\sum_{j\leq n} \exp(-S_{ij})}\right)
    -\log\left(\frac{\exp(-S_{ii})}{\sum_{j\leq n} \exp(-S_{ji})}\right)

We end up choosing the following model output function (check Section 5.1.1 of
our papers for details on why this is a good model output function):

.. math::

    f_{CLIP}(x_i, y_i) =
    -\log\sum_{j\leq n}(\exp(-S_{ii}) - \exp(-S_{ij}))
    -\log\sum_{j\leq n}(\exp(-S_{ii}) - \exp(-S_{ji}))

Because our choice of model output function for CLIP requires access to CLIP
embeddings for multiple examples, we implement an additional utility method
:meth:`.get_embeddings`. This is a bit too specific to CLIP, so we're not going
to pay too much attention to it in this tutorial; let's just assume we have
access to the arrays :code:`all_img_embeddings` and :code:`all_txt_embeddings`.

TODO: @Sam - we need to add a bit more text explaining what's going on
(especially functorch stuff)

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


TODO: @Sam - add a short snippet about
:meth:`.CLIPModelOutput.get_out_to_loss_grad` and the :math:`Q` term.

.. code-block:: python

    def get_out_to_loss_grad(self, func_model, weights, buffers, batch):
        image_embeddings, text_embeddings, temp = func_model(weights, buffers, *batch)
        if self.temperature is None:
            self.temperature = temp
        res = self.temperature * image_embeddings @ text_embeddings.T
        ps = (self.softmax(res) + self.softmax(res.T)).diag() / 2.
        return (1 - ps).clone().detach()

