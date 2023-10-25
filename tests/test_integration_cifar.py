from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import logging
import pytest

from trak import TRAKer
from trak.projectors import BasicProjector


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
                    logging_level=logging.DEBUG,
                    device=device)

    ckpts = [model.state_dict(), model.state_dict()]
    for model_id, ckpt in enumerate(ckpts):
        traker.load_checkpoint(ckpt, model_id=model_id)

        for batch in tqdm(loader_train, desc='Computing TRAK embeddings...'):
            batch = [x.to(device) for x in batch]
            traker.featurize(batch=batch,
                             num_samples=loader_train.batch_size)
            break  # a CPU pass takes too long lol


@pytest.mark.cuda
def test_cifar10_cuda(tmp_path):
    test_cifar10(tmp_path, device='cuda:0')
