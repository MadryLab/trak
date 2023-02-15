import pytest
from traker.traker import TRAKer
from traker.modelout_functions import CrossEntropyModelOutput
from torchvision.models import resnet18

def test_class_init():
    model = resnet18()
    modelout_fn = CrossEntropyModelOutput(device='CPU')
    traker = TRAKer(save_dir='.',
                    model=model)
