.. _SLURM tutorial:

Parallelize :code:`TRAK` scoring with :code:`SLURM`
===================================================

Often we would like to compute :code:`TRAK` scores from multiple checkpoints of
the same model.

.. note::

    Check `our paper <https://arxiv.org/abs/2303.14186>`_ to see why using multiple checkpoints helps improve :code:`TRAK`'s performance.

This means that we need to run :meth:`.TRAKer.featurize` for all
training examples *for each checkpoint*. But fortunately, this is a highly parallelizable
process!

Below, we sketch a simple way of parallelizing :meth:`.featurize` and
:meth:`.score` across checkpoints. We'll use `SLURM
<https://slurm.schedmd.com/overview.html>`_ --- a popular job scheduling
system.

.. note::

    You can find all the code for this example `here
    <https://github.com/MadryLab/trak/tree/main/examples/slurm_example>`_. We'll
    skip some details in the post to highlight the main ideas behind using
    :code:`TRAK` with :code:`SLURM`.

Overall, we'll write three files:

* :code:`featurize_and_score.py`
* :code:`run.sbatch`
* :code:`gather.py`

We will use :code:`run.sbatch` to run different instances of :code:`featurize_and_score.py`
in parallel, and get the final :code:`TRAK` scores using :code:`gather.py`.

.. note::

    In terms of MapReduce, you can of :code:`featurize_and_score` as the map function and :code:`gather` as the reduce function.


1. Featurizing each checkpoint
------------------------------

Everything needed for scoring prior to :meth:`.finalize_scores` will go in
:code:`featurize_and_score.py`.
For example, :code:`featurize_and_score.py` can be as follows:

.. code-block:: python
    :linenos:
    :emphasize-lines: 6,7,16

    from argparse import ArgumentParser
    from trak import TRAKer

    def main(model_id):
        model,loader_train, loader_val = ...
        # use model_id here to load the respective checkpoint, e.g.:
        ckpt = torch.load(f'/path/to/checkpoints/ckpt_{model_id}.pt')

        traker = TRAKer(model=model,
                        task='image_classification',
                        train_set_size=len(ds_train))

        traker.load_checkpoint(ckpt, model_id=model_id)
        for batch in loader_train:
            traker.featurize(batch=batch, ...)
        traker.finalize_features(model_ids=[model_id])

        traker.start_scoring_checkpoint(ckpt, model_id, ...)
        for batch in loader_val:
            traker.score(batch=batch, ...)

        # This will be called from gather.py instead.
        # scores = traker.finalize_scores()

    if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument('--model_id', required=True, type=int)
        args = parser.parse_args()
        main(args.model_id)

2. Run featurize in parallel
----------------------------


Now we can run the above script script in parallel with a :code:`run.sbatch`.
Here is a minimal example:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task=8
    #SBATCH --gres=gpu:a100:1
    #SBATCH --array=0-9
    #SBATCH --job-name=trak

    MODEL_ID=$SLURM_ARRAY_TASK_ID

    python featurize_and_score.py --model_id $MODEL_ID

The above script will submit 10 jobs in parallel or us: this is specified by the
:code:`#SBATCH array=0-9` command. Each job will pass in the job ID as a model
ID for :code:`TRAK`. To learn more about the :code:`SBATCH`, check out
:code:`SLURM`\ s `docs <https://slurm.schedmd.com/sbatch.html>`_.

Note that on line 16 of the example :code:`featurize_and_score.py` above, we
call :meth:`.finalize_features` with :code:`model_ids=[model_id]`. This is
important --- if we don't specify this, :code:`TRAK` by default attempts to
finalize the features for all :code:`model_id`\ s (checkpoints) in the
:code:`save_dir` of the current :class:`.TRAKer` instance.

Running

.. code:: bash

    sbatch run.sbatch

in the terminal will populate the specified :code:`save_dir` with all
intermediate results we need to compute the final :code:`TRAK` scores.

3. Gather final scores
----------------------

The only thing left to do is call :meth:`.TRAKer.finalize_scores`. This method
combines the scores across checkpoints (think of it as a :code:`gather`).
This is what :code:`gather.py` will do:

.. code-block:: python

    from trak import TRAKer

    model = ...

    traker = TRAKer(model=model, task='image_classification', ...)
    scores = traker.finalize_scores()

That's it!

.. note::

    Ease of parallelization was a priority for us when we designed :code:`TRAK`.
    The above example uses :code:`SLURM` to achieve parallelization but is
    definitely not the only option --- for example, you should have no problems
    integrating :code:`TRAK` with `torch distributed
    <https://pytorch.org/docs/stable/notes/ddp.html>`_.