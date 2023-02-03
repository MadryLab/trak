import pytest
from trak.traker import TRAKer
from trak.modelout_functions import CrossEntropyModelOutput
from torchvision.models import resnet18

def test_class_init():
    model = resnet18()
    modelout_fn = CrossEntropyModelOutput(device='CPU')
    traker = TRAKer(save_dir='.',
                    model=model,
                    model_output_fn=modelout_fn)
