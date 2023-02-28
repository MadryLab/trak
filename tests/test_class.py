import pytest
from traker.traker import TRAKer
from torchvision.models import resnet18

def test_class_init():
    model = resnet18()
    traker = TRAKer(model=model,
                    task='image_classification',
                    save_dir='./trak_test_class_results',
                    train_set_size=20,
                    device='cuda:0')