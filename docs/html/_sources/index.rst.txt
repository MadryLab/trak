.. TRAK documentation master file, created by
   sphinx-quickstart on Mon Mar 13 15:39:17 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TRAK's documentation!
================================

.. note::

   :code:`TRAK` is under active development. We are still in a :code:`0.x.x`
   version and lots of things *may* change.


Overview
--------

TODO: add links to paper, blog; a short summary of :code:`TRAK`; figure 1 from
the paper.

Install
-------

The PyTorch-only version of our package can be installed using

.. code-block:: bash

    pip install traker

To install the version of our package which contains a fast, custom CUDA kernel
for the JL projection step, use

.. code-block:: bash

    pip install traker[fast]

See the :doc:`install` for more details.

Contents
--------

.. toctree::
   :maxdepth: 1

   quickstart
   install
   slurm
   clip
   bert
   trak

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
