import pytest
from trak.traker import TRAKer
from trak.model_output_fns import CrossEntropyModelOutput
from torchvision.models import resnet18

def test_class_init():
    model = resnet18()
    model_output_fn = CrossEntropyModelOutput(device='CPU')
    traker = TRAKer(save_dir='.',
                    model=model,
                    model_output_fn=model_output_fn)