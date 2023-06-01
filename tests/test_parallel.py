import pytest
import logging
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch as ch

from trak import TRAKer
from .utils import construct_rn9, get_dataloader, eval_correlations
from .utils import download_cifar_checkpoints, download_cifar_betons


@pytest.mark.cuda
def test_featurize_and_score_in_parallel(tmp_path):
    device = 'cuda:0'
    batch_size = 100

    model = construct_rn9().to(memory_format=ch.channels_last).to(device)
    model = model.eval()

    BETONS_PATH = Path(tmp_path).joinpath('cifar_betons')
    BETONS = download_cifar_betons(BETONS_PATH)

    loader_train = get_dataloader(BETONS, batch_size=batch_size, split='train')
    loader_val = get_dataloader(BETONS, batch_size=batch_size, split='val')

    CKPT_PATH = Path(tmp_path).joinpath('cifar_ckpts')
    ckpt_files = download_cifar_checkpoints(CKPT_PATH, ds='cifar2')
    ckpts = [ch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]

    # this should be essentially equivalent to running each
    # TRAKer in a separate script
    for model_id, ckpt in enumerate(ckpts):
        traker = TRAKer(model=model,
                        task='image_classification',
                        train_set_size=10_000,
                        save_dir=tmp_path,
                        device=device,
                        logging_level=logging.DEBUG)
        traker.load_checkpoint(checkpoint=ckpt, model_id=model_id)
        for batch in tqdm(loader_train, desc='Computing TRAK embeddings...'):
            traker.featurize(batch=batch, num_samples=len(batch[0]))
        traker.finalize_features()

    for model_id, ckpt in enumerate(ckpts):
        traker = TRAKer(model=model,
                        task='image_classification',
                        train_set_size=10_000,
                        save_dir=tmp_path,
                        device=device,
                        logging_level=logging.DEBUG)

        traker.start_scoring_checkpoint('test_experiment', ckpt, model_id, num_targets=2_000)
        for batch in tqdm(loader_val, desc='Scoring...'):
            traker.score(batch=batch, num_samples=len(batch[0]))

    scores = traker.finalize_scores(exp_name='test_experiment')

    avg_corr = eval_correlations(infls=scores, tmp_path=tmp_path, ds='cifar2')
    assert avg_corr > 0.062, 'correlation with 3 CIFAR-2 models should be >= 0.062'


@pytest.mark.cuda
def test_score_multiple(tmp_path):
    device = 'cuda:0'
    batch_size = 100

    model = construct_rn9().to(memory_format=ch.channels_last).to(device)
    model = model.eval()

    BETONS_PATH = Path(tmp_path).joinpath('cifar_betons')
    BETONS = download_cifar_betons(BETONS_PATH)

    loader_train = get_dataloader(BETONS, batch_size=batch_size, split='train')
    loader_val = get_dataloader(BETONS, batch_size=batch_size, split='val')

    CKPT_PATH = Path(tmp_path).joinpath('cifar_ckpts')
    ckpt_files = download_cifar_checkpoints(CKPT_PATH, ds='cifar2')
    ckpts = [ch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]

    traker = TRAKer(model=model,
                    task='image_classification',
                    train_set_size=10_000,
                    save_dir=tmp_path,
                    device=device,
                    logging_level=logging.DEBUG)

    for model_id, ckpt in enumerate(ckpts):
        traker.load_checkpoint(checkpoint=ckpt, model_id=model_id)
        for batch in tqdm(loader_train, desc='Computing TRAK embeddings...'):
            traker.featurize(batch=batch, num_samples=len(batch[0]))
    traker.finalize_features()

    scoring_runs = range(3)
    for _ in scoring_runs:
        for model_id, ckpt in enumerate(ckpts):
            traker = TRAKer(model=model,
                            task='image_classification',
                            train_set_size=10_000,
                            save_dir=tmp_path,
                            device=device,
                            logging_level=logging.DEBUG)

            traker.start_scoring_checkpoint('test_experiment', ckpt, model_id, num_targets=2_000)
            for batch in tqdm(loader_val, desc='Scoring...'):
                traker.score(batch=batch, num_samples=len(batch[0]))

        scores = traker.finalize_scores('test_experiment')

        avg_corr = eval_correlations(infls=scores, tmp_path=tmp_path, ds='cifar2')
        assert avg_corr > 0.062, 'correlation with 3 CIFAR-2 models should be >= 0.062'


@pytest.mark.cuda
def test_score_in_shards(tmp_path):
    device = 'cuda:0'
    batch_size = 100

    model = construct_rn9().to(memory_format=ch.channels_last).to(device)
    model = model.eval()

    BETONS_PATH = Path(tmp_path).joinpath('cifar_betons')
    BETONS = download_cifar_betons(BETONS_PATH)

    loader_train = get_dataloader(BETONS, batch_size=batch_size, split='train')

    CKPT_PATH = Path(tmp_path).joinpath('cifar_ckpts')
    ckpt_files = download_cifar_checkpoints(CKPT_PATH, ds='cifar2')
    ckpts = [ch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]

    traker = TRAKer(model=model,
                    task='image_classification',
                    train_set_size=10_000,
                    save_dir=tmp_path,
                    device=device,
                    logging_level=logging.DEBUG)

    for model_id, ckpt in enumerate(ckpts):
        traker.load_checkpoint(checkpoint=ckpt, model_id=model_id)
        for batch in tqdm(loader_train, desc='Computing TRAK embeddings...'):
            traker.featurize(batch=batch, num_samples=len(batch[0]))
    traker.finalize_features()

    scoring_shards = [np.arange(1000), np.arange(1000, 2000)]
    # this should be essentially equivalent to scoring each
    # shard in a separate script
    for scoring_inds in scoring_shards:
        loader_val = get_dataloader(BETONS, batch_size=batch_size,
                                    split='val', indices=scoring_inds)
        for model_id, ckpt in enumerate(ckpts):
            traker = TRAKer(model=model,
                            task='image_classification',
                            train_set_size=10_000,
                            save_dir=tmp_path,
                            device=device,
                            logging_level=logging.DEBUG)

            traker.start_scoring_checkpoint('test_experiment', ckpt, model_id, num_targets=2000)
            for batch_idx, batch in enumerate(tqdm(loader_val, desc='Scoring...')):
                batch_inds = scoring_inds[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                traker.score(batch=batch, inds=batch_inds)

    scores = traker.finalize_scores('test_experiment')

    avg_corr = eval_correlations(infls=scores, tmp_path=tmp_path, ds='cifar2')
    assert avg_corr > 0.062, 'correlation with 3 CIFAR-2 models should be >= 0.062'


@pytest.mark.cuda
def test_featurize_in_shards(tmp_path):
    device = 'cuda:0'
    batch_size = 100

    model = construct_rn9().to(memory_format=ch.channels_last).to(device)
    model = model.eval()

    BETONS_PATH = Path(tmp_path).joinpath('cifar_betons')
    BETONS = download_cifar_betons(BETONS_PATH)

    loader_val = get_dataloader(BETONS, batch_size=batch_size, split='val')

    CKPT_PATH = Path(tmp_path).joinpath('cifar_ckpts')
    ckpt_files = download_cifar_checkpoints(CKPT_PATH, ds='cifar2')
    ckpts = [ch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]

    # this should be essentially equivalent to featurizing each
    # shard in a separate script
    featurizing_shards = [np.arange(5000), np.arange(5000, 10_000)]
    for featurizing_inds in featurizing_shards:
        loader_train = get_dataloader(BETONS, batch_size=batch_size,
                                      split='train', indices=featurizing_inds)
        traker = TRAKer(model=model,
                        task='image_classification',
                        train_set_size=10_000,
                        save_dir=tmp_path,
                        device=device,
                        logging_level=logging.DEBUG)

        for model_id, ckpt in enumerate(ckpts):
            traker.load_checkpoint(checkpoint=ckpt, model_id=model_id)
            for batch_idx, batch in enumerate(tqdm(loader_train, desc='Computing TRAK embeddings')):
                batch_inds = featurizing_inds[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                traker.featurize(batch=batch, inds=batch_inds)

    traker.finalize_features()

    traker = TRAKer(model=model,
                    task='image_classification',
                    train_set_size=10_000,
                    save_dir=tmp_path,
                    device=device,
                    logging_level=logging.DEBUG)

    for model_id, ckpt in enumerate(ckpts):

        traker.start_scoring_checkpoint('test_experiment', ckpt, model_id, num_targets=2_000)
        for batch in tqdm(loader_val, desc='Scoring...'):
            traker.score(batch=batch, num_samples=len(batch[0]))

    scores = traker.finalize_scores('test_experiment')

    avg_corr = eval_correlations(infls=scores, tmp_path=tmp_path, ds='cifar2')
    assert avg_corr > 0.062, 'correlation with 3 CIFAR-2 models should be >= 0.062'


@pytest.mark.cuda
def test_preemption(tmp_path):
    device = 'cuda:0'
    batch_size = 100

    model = construct_rn9().to(memory_format=ch.channels_last).to(device)
    model = model.eval()

    BETONS_PATH = Path(tmp_path).joinpath('cifar_betons')
    BETONS = download_cifar_betons(BETONS_PATH)

    loader_val = get_dataloader(BETONS, batch_size=batch_size, split='val')

    CKPT_PATH = Path(tmp_path).joinpath('cifar_ckpts')
    ckpt_files = download_cifar_checkpoints(CKPT_PATH, ds='cifar2')
    ckpts = [ch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]

    # this should be essentially equivalent to featurizing each
    # shard in a separate script
    featurizing_shards = [np.arange(5000), np.arange(10_000)]
    for featurizing_inds in featurizing_shards:
        loader_train = get_dataloader(BETONS, batch_size=batch_size,
                                      split='train', indices=featurizing_inds)
        traker = TRAKer(model=model,
                        task='image_classification',
                        train_set_size=10_000,
                        save_dir=tmp_path,
                        device=device,
                        logging_level=logging.DEBUG)

        for model_id, ckpt in enumerate(ckpts):
            traker.load_checkpoint(checkpoint=ckpt, model_id=model_id)
            for batch_idx, batch in enumerate(tqdm(loader_train, desc='Computing TRAK embeddings')):
                batch_inds = featurizing_inds[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                traker.featurize(batch=batch, inds=batch_inds)

    traker.finalize_features()

    traker = TRAKer(model=model,
                    task='image_classification',
                    train_set_size=10_000,
                    save_dir=tmp_path,
                    device=device,
                    logging_level=logging.DEBUG)

    for model_id, ckpt in enumerate(ckpts):

        traker.start_scoring_checkpoint('test_experiment', ckpt, model_id, num_targets=2_000)
        for batch in tqdm(loader_val, desc='Scoring...'):
            traker.score(batch=batch, num_samples=len(batch[0]))

    scores = traker.finalize_scores('test_experiment')

    avg_corr = eval_correlations(infls=scores, tmp_path=tmp_path, ds='cifar2')
    assert avg_corr > 0.062, 'correlation with 3 CIFAR-2 models should be >= 0.062'
