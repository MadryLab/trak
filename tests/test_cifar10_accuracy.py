from tqdm import tqdm
from pathlib import Path
from itertools import product
import pytest
import torch
import torchvision

from trak import TRAKer
from trak.projectors import BasicProjector

from .utils import construct_rn9, download_cifar_checkpoints, eval_correlations


ch = torch


def get_dataloader(batch_size=256, num_workers=8, split='train', shuffle=False, augment=True):
    if augment:
        transforms = torchvision.transforms.Compose(
                        [torchvision.transforms.RandomHorizontalFlip(),
                         torchvision.transforms.RandomAffine(0),
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2023, 0.1994, 0.201))])
    else:
        transforms = torchvision.transforms.Compose([
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2023, 0.1994, 0.201))])

    is_train = (split == 'train')
    dataset = torchvision.datasets.CIFAR10(root='/tmp/cifar/',
                                           download=True,
                                           train=is_train,
                                           transform=transforms)

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         shuffle=shuffle,
                                         batch_size=batch_size,
                                         num_workers=num_workers)

    return loader


def get_projector(use_cuda_projector, dtype):
    if use_cuda_projector:
        return None
    return BasicProjector(grad_dim=2274880, proj_dim=2048,
                          seed=0, proj_type='normal', block_size=400,
                          dtype=dtype, device='cuda:0')


# reduce the number of tests for CIFAR-10
PARAM = list(product([False],  # serialize
                     [True],  # basic / cuda projector
                     [ch.float16],  # projection dtype
                     [128],  # batch size
                     ))


@pytest.mark.parametrize("serialize, use_cuda_projector, dtype, batch_size", PARAM)
@pytest.mark.cuda
def test_cifar_acc(serialize, use_cuda_projector, dtype, batch_size, tmp_path):
    device = 'cuda:0'
    projector = get_projector(use_cuda_projector, dtype)
    model = construct_rn9(10).to(memory_format=ch.channels_last).to(device)
    model = model.eval()

    loader_train = get_dataloader(batch_size=batch_size, split='train', augment=False)
    loader_val = get_dataloader(batch_size=batch_size, split='val', augment=False)

    CKPT_PATH = Path(tmp_path).joinpath('cifar_ckpts')
    ckpt_files = download_cifar_checkpoints(CKPT_PATH)

    ckpts = [ch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]

    traker = TRAKer(model=model,
                    task='image_classification',
                    projector=projector,
                    train_set_size=50_000,
                    save_dir=tmp_path,
                    device=device)

    for model_id, ckpt in enumerate(ckpts):
        traker.load_checkpoint(checkpoint=ckpt, model_id=model_id)
        for batch in tqdm(loader_train, desc='Computing TRAK embeddings...'):
            batch = [x.cuda() for x in batch]
            traker.featurize(batch=batch, num_samples=len(batch[0]))

    traker.finalize_features()

    if serialize:
        del traker
        traker = TRAKer(model=model,
                        task='image_classification',
                        projector=projector,
                        train_set_size=50_000,
                        save_dir=tmp_path,
                        device=device)

    for model_id, ckpt in enumerate(ckpts):
        traker.start_scoring_checkpoint(exp_name='test_experiment',
                                        checkpoint=ckpt,
                                        model_id=model_id,
                                        num_targets=10_000)
        for batch in tqdm(loader_val, desc='Scoring...'):
            batch = [x.cuda() for x in batch]
            traker.score(batch=batch, num_samples=len(batch[0]))

    print(traker.saver.experiments)

    scores = traker.finalize_scores(exp_name='test_experiment')
    print(scores)
    print(scores.shape)

    avg_corr = eval_correlations(infls=scores, tmp_path=tmp_path)
    assert avg_corr > 0.05, 'correlation with the above 3 CIFAR-10 checkpoints should be >= 0.05'
