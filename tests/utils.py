from typing import List
import wget
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr
import numpy as np
import torch
import torchvision
ch = torch

try:
    from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.pipeline.operation import Operation
    from ffcv.transforms import RandomHorizontalFlip, Cutout, \
        RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
    from ffcv.transforms.common import Squeeze
except:
    print('No ffcv installed')

BETONS = {
        'train': "/mnt/cfs/home/spark/cifar_ffcv/cifar2/train.beton",
        'val': "/mnt/cfs/home/spark/cifar_ffcv/cifar2/val.beton",
}

STATS = {
        'mean': [125.307, 122.961, 113.8575],
        'std': [51.5865, 50.847, 51.255]
}

def get_dataloader(batch_size=256,
                   num_workers=8,
                   split='train',  # split \in [train, val]
                   aug_seed=0,
                   should_augment=True,
                   indices=None):
        label_pipeline: List[Operation] = [IntDecoder(),
                                           ToTensor(),
                                           ToDevice(ch.device('cuda:0')),
                                           Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        if should_augment:
                image_pipeline.extend([
                        RandomHorizontalFlip(),
                        RandomTranslate(padding=2, fill=tuple(map(int, STATS['mean']))),
                        Cutout(4, tuple(map(int, STATS['std']))),
                ])

        image_pipeline.extend([
            ToTensor(),
            ToDevice(ch.device('cuda:0'), non_blocking=True),
            ToTorchImage(),
            Convert(ch.float32),
            torchvision.transforms.Normalize(STATS['mean'], STATS['std']),
        ])

        return Loader(BETONS[split],
                      batch_size=batch_size,
                      num_workers=num_workers,
                      order=OrderOption.SEQUENTIAL,
                      drop_last=False,
                      seed=aug_seed,
                      indices=indices,
                      pipelines={'image': image_pipeline, 'label': label_pipeline})

# Resnet9
class Mul(ch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * self.weight


class Flatten(ch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


class Residual(ch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)


def construct_rn9(num_classes=2):
    def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
        return ch.nn.Sequential(
                ch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, groups=groups, bias=False),
                ch.nn.BatchNorm2d(channels_out),
                ch.nn.ReLU(inplace=True)
        )
    model = ch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(ch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        ch.nn.MaxPool2d(2),
        Residual(ch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        ch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        ch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2)
    )
    return model

def eval_correlations(infls, tmp_path):
    masks_url = 'https://www.dropbox.com/s/2nmcjaftdavyg0m/mask.npy?dl=1'
    margins_url = 'https://www.dropbox.com/s/tc3r3c3kgna2h27/val_margins.npy?dl=1'

    masks_path = Path(tmp_path).joinpath('mask.npy')
    wget.download(masks_url, out=str(masks_path), bar=None)
    # num masks, num train samples
    masks = ch.as_tensor(np.load(masks_path, mmap_mode='r')).float()

    margins_path = Path(tmp_path).joinpath('val_margins.npy')
    wget.download(margins_url, out=str(margins_path), bar=None)
    # num , num val samples
    margins = ch.as_tensor(np.load(margins_path, mmap_mode='r'))

    val_inds = np.arange(2000)
    preds = masks @ infls
    rs = []
    ps = []
    for ind, j in tqdm(enumerate(val_inds)):
        r, p = spearmanr(preds[:, ind], margins[:, j])
        rs.append(r)
        ps.append(p)
    rs, ps = np.array(rs), np.array(ps)
    print(f'Correlation: {rs.mean()} (avg p value {ps.mean()})')
    return rs.mean()
