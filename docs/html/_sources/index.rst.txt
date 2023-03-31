.. TRAK documentation master file, created by
   sphinx-quickstart on Mon Mar 13 15:39:17 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TRAK: Attributing Model Behavior at Scale
=========================================

.. note::

   :code:`TRAK` is under active development. We are still in a :code:`0.x.x`
   version and lots of things *may* change.


Overview
--------

This is a `PyTorch <https://pytorch.org/>`_-based API for our method :code:`TRAK`: an
effective, efficient data attribution method for gradient-based learning
algorithms. [1]_ We designed :code:`TRAK`'s API around the following guiding
principles:



Ease of use
   You can apply :code:`TRAK` in just a few lines of code
   (see :doc:`the quickstart guide <quickstart>`).

Speed
   Our API comes with *fast*, custom CUDA kernels. Getting state-of-the-art
   attribution for `BERT-base <https://huggingface.co/bert-base-uncased>`_ on
   `QNLI <https://paperswithcode.com/dataset/qnli>`_ takes ~2 hours on a 8xA100
   node.

Simplicity
   Our API is lightweight - the entire codebase is less than 1000 lines of
   code. It is also quite modular, making it painless adapt any component to
   your needs.

Flexibility
   Applying :code:`TRAK` to a custom task/modality is easy (check, e.g.,
   :doc:`how to adapt TRAK to CLIP<clip>`).

Install
-------

The PyTorch-only version of our package can be installed using

.. code-block:: bash

    pip install traker

To install the version of our package which contains a fast, custom CUDA kernel,
use

.. code-block:: bash

    pip install traker[fast]

See the :doc:`install` for more details.

.. [1] Check `our paper <https://arxiv.org/abs/2303.14186>`_ for more details.

----

Contents
--------

.. toctree::
   :maxdepth: 1

   quickstart
   install
   slurm
   modeloutput
   bert
   clip
   trak

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
