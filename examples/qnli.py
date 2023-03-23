"""
Example applying TRAK to language models finetuned for text classification.

Dataset: GLUE QNLI
Model: bert-base-cased (https://huggingface.co/bert-base-cased)

Tokenizers and loaders are adapted from the Hugging Face example
(https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification).

"""

from argparse import ArgumentParser
import sys
from tqdm import tqdm

import torch as ch
import torch.nn as nn
from torch.utils.data import DataLoader

from trak import TRAKer

# Huggingface
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
)



GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# NOTE: CHANGE THIS IF YOU WANT TO RUN ON FULL DATASET
TRAIN_SET_SIZE = 5_000
VAL_SET_SIZE = 1_00


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


def init_loaders(batch_size=16):
    ds_train = get_dataset('train')
    ds_train = ds_train.select(range(TRAIN_SET_SIZE))
    ds_val = get_dataset('val')
    ds_val = ds_val.select(range(VAL_SET_SIZE))
    return DataLoader(ds_train, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator), \
        DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)


def process_batch(batch):
    return batch['input_ids'], batch['token_type_ids'], batch['attention_mask'], batch['labels']


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str, help='model checkpoint', required=True)
    parser.add_argument('--out', type=str, help='dir to save TRAK scores and metadata to', required=True)
    args = parser.parse_args()

    device = 'cuda'
    loader_train, loader_val = init_loaders()
    model = init_model(args.ckpt, device)

    traker = TRAKer(model=model,
                    task='text_classification',
                    train_set_size=TRAIN_SET_SIZE,
                    save_dir=args.out,
                    device=device,
                    proj_dim=1024)

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