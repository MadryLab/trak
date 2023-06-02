.. _BERT tutorial:

Applying :code:`TRAK` to a custom task #2: Text Classification using BERT
=======================================================================================

In this tutorial, we'll show another example of applying :code:`TRAK` to a new
custom task, text classification. If you haven't,
you should first check out :ref:`MODELOUTPUT tutorial` to familiarize yourself with the notion of
a model output function and how we implement it inside :code:`TRAK`.
Adapting to text classification is pretty simple as the task at hand is still classification.

We will use a pre-trained langauge model (`bert-base-cased <https://huggingface.co/bert-base-cased>`_) from HuggingFace and finetune it on
the `QNLI (Question-answering NLI) task <https://huggingface.co/datasets/SetFit/qnli>`_, which is a binary classification task.
You can find the end-to-end example `here <https://github.com/MadryLab/trak/blob/main/examples/qnli.py>`_.


Model and Data
-------------------------

For the model architecture, we adapt :code:`transformers.AutoModelForSequenceClassification`
to fit our API signatures.

.. code-block:: python

    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
    )

    class SequenceClassificationModel(nn.Module):
        """
        Wrapper for HuggingFace sequence classification models.
        """
        def __init__(self):
            super().__init__()
            self.config = AutoConfig.from_pretrained(
                'bert-base-cased',
                num_labels=2,
                finetuning_task='qnli',
                cache_dir=None,
                revision='main',
                use_auth_token=None,
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                'bert-base-cased',
                config=self.config,
                cache_dir=None,
                revision='main',
                use_auth_token=None,
                ignore_mismatched_sizes=False
            )
            self.model.eval().cuda()

        def forward(self, input_ids, token_type_ids, attention_mask):
            return self.model(input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask).logits

    model = SequenceClassificationModel()
    sd = ch.load(CKPT_PATH)
    model.model.load_state_dict(sd)

We slightly redefine the :code:`forward` function so that we can pass in the inputs (:code:`input_ids`, etc.) as positional arguments instead of as keyword arguments.

For data loading, we adapt the code from Hugging Face example:

.. raw:: html

    <details>
    <summary><a>Data loading code for QNLI </a></summary>

.. code-block:: python

    def get_dataset(split, inds=None):
        raw_datasets = load_dataset(
                "glue",
                'qnli',
                cache_dir=None,
                use_auth_token=None,
            )
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
        sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS['qnli']

        label_to_id = None #{v: i for i, v in enumerate(label_list)}

        tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-cased',
            cache_dir=None,
            use_fast=True,
            revision='main',
            use_auth_token=False
        )

        padding = "max_length"
        max_seq_length=128

        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
            return result

        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=(not False),
            desc="Running tokenizer on dataset",
        )

        if split == 'train':
            train_dataset = raw_datasets["train"]
            ds = train_dataset
        else:
            eval_dataset = raw_datasets["validation"]
            ds = eval_dataset
        return ds


    def init_model(ckpt_path, device='cuda'):
        model = SequenceClassificationModel()
        sd = ch.load(ckpt_path)
        model.model.load_state_dict(sd)
        return model

    # NOTE: CHANGE THIS IF YOU WANT TO RUN ON FULL DATASET
    TRAIN_SET_SIZE = 5_000
    VAL_SET_SIZE = 1_00

    def init_loaders(batch_size=16):
        ds_train = get_dataset('train')
        ds_train = ds_train.select(range(TRAIN_SET_SIZE))
        ds_val = get_dataset('val')
        ds_val = ds_val.select(range(VAL_SET_SIZE))
        return DataLoader(ds_train, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator), \
            DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)

    loader_train, loader_val = init_loaders()

.. raw:: html

   </details>



Text Classification
--------------------------

QNLI is a binary classifciation task. Hence, we can use the same model output function we used in
:ref:`MODELOUTPUT tutorial`:

.. math::

    f(z;\theta) = \log\left(\frac{p(z;\theta)}{1 - p(z;\theta)}\right)

where :math:`p(z;\theta)` is the soft-max probability associated for the correct class for example :math:`z`. (See our paper for an explanation of why this is an appropriate choice.)

The corresponding gradient of the loss w.r.t. the model output is, again, given by:

.. math::

    \frac{\partial \ell(z;\theta)}{\partial f} = \frac{\partial}{\partial f}
    \log(1 + \exp(-f)) = -\frac{\exp(-f)}{1 + \exp(-f)}  = -(1 - p(z;\theta))


Implementing the model output function
-------------------------------------------------

For text classification tasks, :code:`TRAK` provides a default implementation of model output function
as :class:`.TextClassificationModelOutput`.
Below, we reproduce the implementation so that you can see how it's implemented.
The model output function is implemented as follows:

.. code-block:: python

    def get_output(func_model,
                   weights: Iterable[Tensor],
                   buffers: Iterable[Tensor],
                   input_id: Tensor,
                   token_type_id: Tensor,
                   attention_mask: Tensor,
                   label: Tensor,
                   ) -> Tensor:
        logits = func_model(weights, buffers, input_id.unsqueeze(0),
                                token_type_id.unsqueeze(0),
                                attention_mask.unsqueeze(0))
        bindex = ch.arange(logits.shape[0]).to(logits.device, non_blocking=False)
        logits_correct = logits[bindex, label.unsqueeze(0)]

        cloned_logits = logits.clone()
        cloned_logits[bindex, label.unsqueeze(0)] = ch.tensor(-ch.inf).to(logits.device)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins.sum()

The implementation is identical to the standard classification example in :ref:`MODELOUTPUT tutorial`,
except here the signature of the method and the :code:`func_model` is slightly different
as the language model takes in three inputs instead of just one.

Similarly, the gradient function is implemented as follows:

.. code-block:: python

    def get_out_to_loss_grad(self, func_model, weights, buffers, batch: Iterable[Tensor]) -> Tensor:
        input_ids, token_type_ids, attention_mask, labels = batch
        logits = func_model(weights, buffers, input_ids, token_type_ids, attention_mask)
        ps = self.softmax(logits / self.loss_temperature)[ch.arange(logits.size(0)), labels]
        return (1 - ps).clone().detach().unsqueeze(-1)

Putting it together
-------------------

Using the above :code:`TextClassificationModelOutput` implementation, we can compute :code:`TRAK` scores as follows:

.. code-block:: python

    traker = TRAKer(model=model,
                    task=TextClassificationModelOutput, # you can also just pass in "text_classification"
                    train_set_size=TRAIN_SET_SIZE,
                    save_dir=args.out,
                    device=device,
                    proj_dim=1024)

    def process_batch(batch):
        return batch['input_ids'], batch['token_type_ids'], batch['attention_mask'], batch['labels']

    traker.load_checkpoint(model.state_dict(), model_id=0)
    for batch in tqdm(loader_train, desc='Featurizing..'):
        # process batch into compatible form for TRAKer TextClassificationModelOutput
        batch = process_batch(batch)
        batch = [x.cuda() for x in batch]
        traker.featurize(batch=batch, num_samples=batch[0].shape[0])

    traker.finalize_features()

    traker.start_scoring_checkpoint(model.state_dict(), model_id=0, num_targets=VAL_SET_SIZE)
    for batch in tqdm(loader_val, desc='Scoring..'):
        batch = process_batch(batch)
        batch = [x.cuda() for x in batch]
        traker.score(batch=batch, num_samples=batch[0].shape[0])

    scores = traker.finalize_scores()

We use :code:`process_batch` to transform the batch from dictionary (which is the form used by Hugging Face dataloaders) to a tuple.

That's all! You can find this tutorial as a complete script in `here <https://github.com/MadryLab/trak/blob/main/examples/qnli.py>`_.


Extending to other tasks
----------------------------------

For a more involved example that is *not* classification, see :ref:`CLIP tutorial`.