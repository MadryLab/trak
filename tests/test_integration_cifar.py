import pytest
import torch as ch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from traker.projectors import BasicProjector
from traker.traker import TRAKer

def test_cifar10(tmp_path, device='cpu'):
    model = models.resnet18(weights='DEFAULT')
    model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds_train = datasets.CIFAR10(root='/tmp', download=True, train=True, transform=transform)
    loader_train = DataLoader(ds_train, batch_size=10, shuffle=False)
    if device == 'cpu':
        # the default CudaProjector does not work on cpu
        projector = BasicProjector(grad_dim=11689512,
                                   proj_dim=20,
                                   seed=0,
                                   proj_type='rademacher',
                                   device=device)
    else:
        projector = None 
    trak = TRAKer(model=model,
                  task='image_classification',
                  train_set_size=50_000,
                  projector=projector,
                  save_dir=tmp_path,
                  device=device)

    ckpts = [model.state_dict(), model.state_dict()]
    for model_id, ckpt in enumerate(ckpts):
        trak.load_checkpoint(ckpt, model_id=model_id)

        counter = 0
        for batch in tqdm(loader_train, desc='Computing TRAK embeddings...'):
            batch = [x.to(device) for x in batch]
            trak.featurize(batch=batch,
                           num_samples=loader_train.batch_size)
            counter += 1
            if counter == 2:
                break # a CPU pass takes too long lol
    
    trak.finalize_features()

    ds_val = datasets.CIFAR10(root='/tmp', download=True, train=False, transform=transform)
    loader_val = DataLoader(ds_val, batch_size=10, shuffle=False)
    for model_id, ckpt in enumerate(ckpts):
        counter = 0
        trak.load_checkpoint(ckpt, model_id=model_id)
        for batch in tqdm(loader_val, desc='Scoring...'):
            batch = [x.to(device) for x in batch]
            trak.score(batch=batch,
                       num_samples=loader_val.batch_size)
            counter += 1
            if counter == 2:
                break # a CPU pass takes too long lol

    trak.finalize_scores()


@pytest.mark.cuda
def test_cifar10_cuda(tmp_path):
    test_cifar10(tmp_path, device='cuda:0')


def test_cifar10_iter(tmp_path, device='cpu'):
    model = models.resnet18(weights='DEFAULT')
    model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds_train = datasets.CIFAR10(root='/tmp', download=True, train=True, transform=transform)
    loader_train = DataLoader(ds_train, batch_size=10, shuffle=False)

    if device == 'cpu':
        # the default CudaProjector does not work on cpu
        projector = BasicProjector(grad_dim=11689512,
                                   proj_dim=20,
                                   seed=0,
                                   proj_type='rademacher',
                                   device=device)
    else:
        projector = None 

    trak = TRAKer(model=model,
                  task='image_classification',
                  train_set_size=50_000,
                  save_dir=tmp_path,
                  projector=projector,
                  functional=False,
                  device=device)

    trak.load_checkpoint(model.state_dict(), model_id=0)
    counter = 0
    for batch in tqdm(loader_train, desc='Computing TRAK embeddings...'):
        batch = [x.to(device) for x in batch]
        trak.featurize(batch=batch,
                       num_samples=loader_train.batch_size)
        counter += 1
        if counter == 2:
            break # a CPU pass takes too long lol
    
    trak.finalize_features()

    ds_val = datasets.CIFAR10(root='/tmp', download=True, train=False, transform=transform)
    loader_val = DataLoader(ds_val, batch_size=10, shuffle=False)
    # load margins
    counter = 0
    for batch in tqdm(loader_val, desc='Scoring...'):
        batch = [x.to(device) for x in batch]
        s = trak.score(batch=batch, num_samples=loader_val.batch_size)
        counter += 1
        if counter == 2:
            break

@pytest.mark.cuda
def test_cifar10_iter_cuda(tmp_path):
    test_cifar10_iter(tmp_path, device='cuda:0')
