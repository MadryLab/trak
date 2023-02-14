import pytest
import torch as ch
from tqdm import tqdm
from functorch import make_functional_with_buffers
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from traker.traker import TRAKer
from traker.modelout_functions import CrossEntropyModelOutput

def test_cifar10(device='cpu'):
    # TODO: load CIFAR-10 weights instead ('DEFAULT' loads ImageNet ones)
    model = models.resnet18(weights='DEFAULT')
    model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds_train = datasets.CIFAR10(root='/tmp', download=True, train=True, transform=transform)
    loader_train = DataLoader(ds_train, batch_size=64, shuffle=False)

    modelout_fn = CrossEntropyModelOutput(device=device)
    traker = TRAKer(model=model,
                    train_set_size=50_000,
                    grad_dtype=ch.float32,
                    device=device)

    func_model, weights, buffers = make_functional_with_buffers(model)
    def compute_outputs(weights, buffers, image, label):
        # we are only allowed to pass in tensors to vmap,
        # thus func_model is used from above
        out = func_model(weights, buffers, image.unsqueeze(0))
        return modelout_fn.get_output(out, label.unsqueeze(0)).sum()

    def compute_out_to_loss(weights, buffers, images, labels):
        out = func_model(weights, buffers, images)
        return modelout_fn.get_out_to_loss(out, labels)

    ckpts = [None, None]
    for model_id, ckpt in enumerate(ckpts):
        # load state dict here if we actually had checkpoints
        # model.load_state_dict(ckpt)
        # update weights, buffers; etc
        for bind, batch in enumerate(tqdm(loader_train, desc='Computing TRAK embeddings...')):
            batch = [x.to(device) for x in batch]
            inds = list(range(bind * loader_train.batch_size,
                            (bind + 1) * loader_train.batch_size))
            traker.featurize(out_fn=compute_outputs,
                            loss_fn=compute_out_to_loss,
                            model=(func_model, weights, buffers),
                            batch=batch,
                            functional=True,
                            model_id=model_id,
                            inds=inds)
            if bind == 5:
                break # a CPU pass takes too long lol
    
    traker.finalize()

    ds_val = datasets.CIFAR10(root='/tmp', download=True, train=False, transform=transform)
    loader_val = DataLoader(ds_val, batch_size=10, shuffle=False)
    # load margins
    for bind, batch in enumerate(tqdm(loader_val, desc='Scoring...')):
        batch = [x.to(device) for x in batch]
        traker.score(out_fn=compute_outputs, batch=batch,
                     model=(func_model, weights, buffers))
        if bind == 5:
            break


@pytest.mark.cuda
def test_cifar10_cuda():
    test_cifar10(device='cuda:0')


def test_cifar10_iter(device='cpu'):
    # TODO: load CIFAR-10 weights instead ('DEFAULT' loads ImageNet ones)
    model = models.resnet18(weights='DEFAULT')
    model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds_train = datasets.CIFAR10(root='/tmp', download=True, train=True, transform=transform)
    loader_train = DataLoader(ds_train, batch_size=10, shuffle=False)

    modelout_fn = CrossEntropyModelOutput(device=device)
    traker = TRAKer(model=model,
                    train_set_size=50_000,
                    grad_dtype=ch.float32,
                    device=device)

    def compute_outputs(model, images, labels):
        # we are only allowed to pass in tensors to vmap,
        # thus func_model is used from above
        out = model(images)
        return modelout_fn.get_output(out, labels)

    def compute_out_to_loss(model, images, labels):
        out = model(images)
        return modelout_fn.get_out_to_loss(out, labels)

    for bind, batch in enumerate(tqdm(loader_train, desc='Computing TRAK embeddings...')):
        batch = [x.to(device) for x in batch]
        inds = list(range(bind * loader_train.batch_size,
                          (bind + 1) * loader_train.batch_size))
        traker.featurize(out_fn=compute_outputs,
                         loss_fn=compute_out_to_loss,
                         model=model,
                         batch=batch,
                         functional=False,
                         inds=inds)
        if bind == 5:
            break # a CPU pass takes too long lol
    
    traker.finalize()

    ds_val = datasets.CIFAR10(root='/tmp', download=True, train=False, transform=transform)
    loader_val = DataLoader(ds_val, batch_size=10, shuffle=False)
    # load margins
    for bind, batch in enumerate(tqdm(loader_val, desc='Scoring...')):
        batch = [x.to(device) for x in batch]
        s = traker.score(out_fn=compute_outputs, batch=batch,
                         model=model, functional=False)
        if bind == 5:
            break

@pytest.mark.cuda
def test_cifar10_iter_cuda():
    test_cifar10_iter(device='cuda:0')
