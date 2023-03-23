import pytest
from tqdm import tqdm
from pathlib import Path
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
    ckpt_files = download_cifar_checkpoints(CKPT_PATH)
    ckpts = [ch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]

    # this should be essentially equivalent to running each
    # TRAKer in a separate script
    for model_id, ckpt in enumerate(ckpts):
        traker = TRAKer(model=model,
                        task='image_classification',
                        train_set_size=10_000,
                        save_dir=tmp_path,
                        device=device)
        traker.load_checkpoint(checkpoint=ckpt, model_id=model_id)
        for batch in tqdm(loader_train, desc='Computing TRAK embeddings...'):
            traker.featurize(batch=batch, num_samples=len(batch[0]))
        traker.finalize_features()

    for model_id, ckpt in enumerate(ckpts):
        traker = TRAKer(model=model,
                        task='image_classification',
                        train_set_size=10_000,
                        save_dir=tmp_path,
                        device=device)

        traker.start_scoring_checkpoint(ckpt, model_id, num_targets=2_000)
        for batch in tqdm(loader_val, desc='Scoring...'):
            traker.score(batch=batch, num_samples=len(batch[0]))

    scores = traker.finalize_scores().cpu()

    avg_corr = eval_correlations(infls=scores, tmp_path=tmp_path)
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
    ckpt_files = download_cifar_checkpoints(CKPT_PATH)
    ckpts = [ch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]

    traker = TRAKer(model=model,
                    task='image_classification',
                    train_set_size=10_000,
                    save_dir=tmp_path,
                    device=device)

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
                            device=device)

            traker.start_scoring_checkpoint(ckpt, model_id, num_targets=2_000)
            for batch in tqdm(loader_val, desc='Scoring...'):
                traker.score(batch=batch, num_samples=len(batch[0]))

        scores = traker.finalize_scores().cpu()

        avg_corr = eval_correlations(infls=scores, tmp_path=tmp_path)
        assert avg_corr > 0.062, 'correlation with 3 CIFAR-2 models should be >= 0.062'
