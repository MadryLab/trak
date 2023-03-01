import time
import pytest
import shutil
from pathlib import Path
from traker.traker import TRAKer
from torchvision.models import resnet18

def test_class_init():
    model = resnet18()
    traker = TRAKer(model=model,
                    task='image_classification',
                    save_dir='./trak_test_class_results',
                    train_set_size=20,
                    device='cuda:0')
    
def test_load_ckpt():
    model = resnet18()
    traker = TRAKer(model=model,
                    task='image_classification',
                    save_dir='./trak_test_class_results',
                    train_set_size=20,
                    device='cuda:0')
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)

def test_load_ckpt_repeat():
    model = resnet18()
    traker = TRAKer(model=model,
                    task='image_classification',
                    save_dir='./trak_test_class_results_2',
                    train_set_size=20,
                    device='cuda:0')
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.load_checkpoint(ckpt, model_id=1)