import pytest
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from trak import TRAKer
from trak.projectors import BasicProjector
from trak.gradient_computers import IterativeGradientComputer


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
        projector = BasicProjector(grad_dim=sum(x.numel() for x in model.parameters()),
                                   proj_dim=20,
                                   seed=0,
                                   proj_type='rademacher',
                                   device=device)
    else:
        projector = None
    traker = TRAKer(model=model,
                    task='image_classification',
                    train_set_size=len(ds_train),
                    projector=projector,
                    save_dir=tmp_path,
                    device=device)

    ckpts = [model.state_dict(), model.state_dict()]
    for model_id, ckpt in enumerate(ckpts):
        traker.load_checkpoint(ckpt, model_id=model_id)

        for batch in tqdm(loader_train, desc='Computing TRAK embeddings...'):
            batch = [x.to(device) for x in batch]
            traker.featurize(batch=batch,
                             num_samples=loader_train.batch_size)
            break  # a CPU pass takes too long lol

    traker.finalize_features()

    ds_val = datasets.CIFAR10(root='/tmp', download=True, train=False, transform=transform)
    loader_val = DataLoader(ds_val, batch_size=10, shuffle=False)
    for model_id, ckpt in enumerate(ckpts):
        traker.start_scoring_checkpoint(ckpt, model_id, num_targets=100)
        for batch in tqdm(loader_val, desc='Scoring...'):
            batch = [x.to(device) for x in batch]
            traker.score(batch=batch,
                         num_samples=loader_val.batch_size)
            break  # a CPU pass takes too long lol

    traker.finalize_scores()


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
        projector = BasicProjector(grad_dim=sum(x.numel() for x in model.parameters()),
                                   proj_dim=20,
                                   seed=0,
                                   proj_type='rademacher',
                                   device=device)
    else:
        projector = None

    traker = TRAKer(model=model,
                    task='image_classification',
                    train_set_size=len(ds_train),
                    save_dir=tmp_path,
                    projector=projector,
                    gradient_computer=IterativeGradientComputer,
                    device=device)

    traker.load_checkpoint(model.state_dict(), model_id=0)
    for batch in tqdm(loader_train, desc='Computing TRAK embeddings...'):
        batch = [x.to(device) for x in batch]
        traker.featurize(batch=batch,
                         num_samples=loader_train.batch_size)
        break  # a CPU pass takes too long lol

    traker.finalize_features()

    ds_val = datasets.CIFAR10(root='/tmp', download=True, train=False, transform=transform)
    loader_val = DataLoader(ds_val, batch_size=10, shuffle=False)
    # load margins
    traker.start_scoring_checkpoint(model.state_dict(), model_id=0, num_targets=100)
    for batch in tqdm(loader_val, desc='Scoring...'):
        batch = [x.to(device) for x in batch]
        traker.score(batch=batch, num_samples=loader_val.batch_size)
        break


@pytest.mark.cuda
def test_cifar10_iter_cuda(tmp_path):
    test_cifar10_iter(tmp_path, device='cuda:0')
