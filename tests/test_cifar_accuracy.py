from tqdm import tqdm
from pathlib import Path
from itertools import product
import pytest
import torch as ch

from trak import TRAKer
from trak.projectors import BasicProjector

from .utils import construct_rn9, get_dataloader, eval_correlations


def get_projector(use_cuda_projector, dtype):
    if use_cuda_projector:
        return None
    return BasicProjector(grad_dim=2273856, proj_dim=1024,
                          seed=0, proj_type='rademacher',
                          dtype=dtype, device='cuda:0')


PARAM = list(product([False, True],  # serialize
                     [False, True],  # basic / cuda projector
                     [ch.float16, ch.float32],  # projection dtype
                     [100, 32],  # batch size
                     ))


@pytest.mark.parametrize("serialize, use_cuda_projector, dtype, batch_size", PARAM)
@pytest.mark.cuda
def test_cifar_acc(serialize, use_cuda_projector, dtype, batch_size, tmp_path):
    device = 'cuda:0'
    projector = get_projector(use_cuda_projector, dtype)
    model = construct_rn9().to(memory_format=ch.channels_last).to(device)
    model = model.eval()

    loader_train = get_dataloader(batch_size=batch_size, split='train')
    loader_val = get_dataloader(batch_size=batch_size, split='val')

    # TODO: put this on dropbox as well
    CKPT_PATH = '/mnt/xfs/projects/trak/checkpoints/resnet9_cifar2/debug'
    ckpt_files = list(Path(CKPT_PATH).rglob("*.pt"))
    ckpts = [ch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]

    traker = TRAKer(model=model,
                    task='image_classification',
                    projector=projector,
                    proj_dim=1024,
                    train_set_size=10_000,
                    save_dir=tmp_path,
                    device=device)

    for model_id, ckpt in enumerate(ckpts):
        traker.load_checkpoint(checkpoint=ckpt, model_id=model_id)
        for batch in tqdm(loader_train, desc='Computing TRAK embeddings...'):
            traker.featurize(batch=batch, num_samples=len(batch[0]))

    traker.finalize_features()

    if serialize:
        del traker
        traker = TRAKer(model=model,
                        task='image_classification',
                        projector=projector,
                        proj_dim=1024,
                        train_set_size=10_000,
                        save_dir=tmp_path,
                        device=device)

    for model_id, ckpt in enumerate(ckpts):
        traker.start_scoring_checkpoint(ckpt, model_id, num_targets=2_000)
        for batch in tqdm(loader_val, desc='Scoring...'):
            traker.score(batch=batch, num_samples=len(batch[0]))

    scores = traker.finalize_scores().cpu()

    avg_corr = eval_correlations(infls=scores, tmp_path=tmp_path)
    assert avg_corr > 0.058, 'correlation with 3 CIFAR-2 models should be >= 0.058'
