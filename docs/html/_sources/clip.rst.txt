.. _CLIP model output:

Add  a :code:`task` to :code:`TRAKer` (subclassing :code:`ModelOutput`\ ) --- CLIP
==================================================================================



Model output funcitons
----------------------

Computing :code:`TRAK` attribution scores hinges on defining a model output
function to guide the scoring process. We derive and discuss model output
functions for multiple tasks (binary and multiclass classification, CLIP loss,
various NLP tasks, etc) in detail in `our paper <link:TODO>`_. In short, for a
given train set :math:`S`, target example :math:`z` and model parameters
:math:`\theta`, we define