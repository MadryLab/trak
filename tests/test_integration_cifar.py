import pytest
import torch as ch
from tqdm import tqdm
from functorch import make_functional_with_buffers
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from trak.traker import TRAKer
from trak.modelout_functions import CrossEntropyModelOutput

def test_cifar10():
    # TODO: load CIFAR-10 weights instead ('DEFAULT' loads ImageNet ones)
    model = models.resnet18(weights='DEFAULT')
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds_train = datasets.CIFAR10(root='/tmp', download=True, train=True, transform=transform)
    loader_train = DataLoader(ds_train, batch_size=10, shuffle=False)

    modelout_fn = CrossEntropyModelOutput(device='cpu')
    traker = TRAKer(model=model,
                    model_output_fn=modelout_fn,
                    train_set_size=50_000)

    func_model, weights, buffers = make_functional_with_buffers(model)
    def compute_outputs(weights, buffers, image, label):
        # we are only allowed to pass in tensors to vmap,
        # thus func_model is used from above
        out = func_model(weights, buffers, image.unsqueeze(0))
        # return out.sum()
        return modelout_fn.get_output(out, label.unsqueeze(0))

    for bind, batch in enumerate(tqdm(loader_train, desc='Computing TRAK embeddings...')):
        inds = list(range(bind * loader_train.batch_size,
                          (bind + 1) * loader_train.batch_size))
        traker.featurize(compute_outputs, (func_model, weights, buffers),
                         batch, functional=True, inds=inds)
        if bind == 50:
            break
    
    traker.finalize()

    # ds_val = datasets.CIFAR10(root='/tmp', download=True, train=False)
    # loader_val = DataLoader(ds_val, batch_size=256)
    # load margins
