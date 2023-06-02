.. _MODELOUTPUT tutorial:

Applying :code:`TRAK` to a custom task #1: Classification
=======================================================================================

In this tutorial, we'll demonstrate how to apply :code:`TRAK` to a
custom task, using classification as an example.

Applying :code:`TRAK` to a new task requires defining an appropriate **model output function**,
which is implemented by extending :class:`.AbstractModelOutput`.
First, we'll conceptually go over what a model output function is. Then, we will see how it is implemented inside :code:`TRAK`
for the case of (image) classification.

The :code:`TRAK` library already ships with an implementation of :class:`.AbstractModelOutput` for several standard tasks. For example, to use the one corresponding to standard classification (for tasks with a single input, e.g., image classification),
you simply specify the task as follows:

.. code-block:: python

    traker = TRAKer(..., task="image_classification")


Prelim: Model output functions
--------------------------------

Computing :code:`TRAK` scores requires specifying a  **model output function** that you want to attribute. Intuitively, you can just think of it as a some kind of loss or scoring function evaluated on an example.

More formally, given:

* an example of interest :math:`z` (e.g., an input-label pair) and
* model parameters :math:`\theta`,

the model output function :math:`f(z;\theta)` computes a real number based on evaluating the
model on example :math:`z`.

For example, one choice of model output function could be the *loss* :math:`L(z)`
that the model incurs on example :math:`z` (e.g., the cross-entropy loss).
We motivate and derive appropriate model output
functions for several standard tasks (binary and multiclass classification, CLIP loss,
and some NLP tasks) in detail in `our paper <https://arxiv.org/abs/2303.14186>`_.

Give a model output function :math:`f(\cdot;\theta)` and a target example :math:`z` of interest, :code:`TRAK` computes the *attribution score* of each training example :math:`z_i` indicating its importance to :math:`f(z;\theta)`.

Implementing model output functions in :code:`TRAK`
-------------------------------------------------------

In order for :class:`.TRAKer` to compute attribution scores, it needs access to the following two functions:

* The model output function itself, i.e., :math:`f(z;\theta)`
* The gradient of the (training) loss w.r.t. to the model output function, i.e., :math:`\frac{\partial L(z;\theta)}{\partial f}`. We refer to this function simply as *output-to-loss gradient.*

We provide a dedicated class, :class:`.AbstractModelOutput`, that computes the above two functions from a model (a :code:`torch.Module` instance) using the following two functions:

* :meth:`.AbstractModelOutput.get_output`
* :meth:`.AbstractModelOutput.get_out_to_loss_grad`

The :meth:`.AbstractModelOutput.get_output` method implements the model output
function: given a batch of examples, it returns a
vector containing the model outputs for each example in the batch.
This is the
function that :class:`.TRAKer` computes gradients of.

The :meth:`.AbstractModelOutput.get_out_to_loss_grad` method implements the output-to-loss gradient. Since for all the examples in our paper we
could analytically derive this term, we "hardcode"
this in the :code:`get_out_to_loss_grad` method, thus avoiding an additional
gradient computation.

.. note::

    If you find yourself in the (likely rare) situation where you can't
    analytically derive the output-to-loss gradient, you can implement :meth:`.AbstractModelOutput.get_out_to_loss_grad` by
    first computing the model output as in :meth:`.AbstractModelOutput.get_output` and using :code:`autograd` to compute the output-to-loss gradient.

So to apply :code:`TRAK` to a new task, all you have to do is extend :class:`.AbstractModelOutput`
and implement the above two functions, then pass in the new model output object as
the :code:`task` when instantiating :class:`.TRAKer`:

.. code-block:: python

    class CustomModelOutput(AbstractModelOutput):
        def get_output(...):
            # Implement

        def forward(...):
            # Implement

        def get_out_to_loss_grad(...):
            # Implement

    traker = TRAKer(model=model,
                    task=CustomModelOutput,
                    ...)

.. note::

    If you implement a :class:`.AbstractModelOutput` for a common task or objective that you think may be useful to others, please make a pull request
    and we can include it as a default (so that you can just specify the :code:`task` as a string).


Example: Classification
--------------------------------------------------

To illustrate how to implement :class:`.AbstractModelOutput`,  we'll look at the example of standard classification, where the model is optimized to minimize
the cross-entropy loss:

.. math::

    L(z;\theta) = \log(p(z;\theta))

where :math:`p(z;\theta)` is the soft-max probability associated for the correct class :math:`y` for example :math:`z=(x,y)`.

For classification, we use the following model output function:

.. math::

    f(z;\theta) = \log\left(\frac{p(z;\theta)}{1 - p(z;\theta)}\right)

.. note::

    This is the natural analog to the logit function in binary logistic regression. See Section 3 in our paper for an explanation of why this is an appropriate choice.

The corresponding output-to-loss gradient is given by:

.. math::

    \frac{\partial L(z;\theta)}{\partial f} = \frac{\partial}{\partial f}
    \log(1 + \exp(-f)) = -\frac{\exp(-f)}{1 + \exp(-f)}  = -(1 - p(z;\theta))


Implementation
~~~~~~~~~~~~~~~~~

For the above choice of model output function, :code:`TRAK` provides a default implementation
as :class:`.ImageClassificationModelOutput`.
Below, we reproduce the implementation so that you can see how it's implemented.
The model output function is implemented as follows:

.. code-block:: python

    def get_output(model: Module,
                   weights: Iterable[Tensor],
                   buffers: Iterable[Tensor],
                   image: Tensor,
                   label: Tensor):
        logits = ch.func.functional_call(model, (weights, buffers), image.unsqueeze(0))
        bindex = ch.arange(logits.shape[0]).to(logits.device, non_blocking=False)
        logits_correct = logits[bindex, label.unsqueeze(0)]

        cloned_logits = logits.clone()
        # remove the logits of the correct labels from the sum
        # in logsumexp by setting to -ch.inf
        cloned_logits[bindex, label.unsqueeze(0)] = ch.tensor(-ch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins.sum()


Note that the :code:`get_output` function uses :code:`torch.func`'s
:code:`functional_call` to make a stateless forward pass.

.. note::

    In :code:`TRAK`, we use :code:`torch.func`'s :code:`vmap` to make the per-sample gradient
    computations faster. Check out, e.g., `this torch.func tutorial
    <https://pytorch.org/docs/stable/func.whirlwind_tour.html>`_ to
    learn more about how to use :code:`torch.func`.

Similarly, the output-to-loss gradient function is implemented as follows:

.. code-block:: python

    def get_out_to_loss_grad(self, model, weights, buffers, batch):
        images, labels = batch
        logits = ch.func.functional_call(model, (weights, buffers), images)
        # here we are directly implementing the gradient instead of relying on autodiff to do
        # that for us
        ps = self.softmax(logits / self.loss_temperature)[ch.arange(logits.size(0)), labels]
        return (1 - ps).clone().detach().unsqueeze(-1)

Note that we are directly implementing the gradient we analytically derived above (instead of using automatic differentiation).

That's all!
Though we showed how :class:`.ImageClassificationModelOutput` is implemented inside, to use it you just need to specify
:code:`task=image_classification` when instantiating :class:`.TRAKer`.

Extending to other tasks
----------------------------------
For more examples, see :ref:`BERT tutorial` and :ref:`CLIP tutorial`.