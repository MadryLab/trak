import time
import pytest
import shutil
from pathlib import Path
from traker.traker import TRAKer
from torchvision.models import resnet18
import torch as ch

def test_class_init(tmp_path):
    model = resnet18()
    traker = TRAKer(model=model,
                    task='image_classification',
                    save_dir=tmp_path,
                    train_set_size=20,
                    device='cuda:0')
    
def test_load_ckpt(tmp_path):
    model = resnet18()
    traker = TRAKer(model=model,
                    task='image_classification',
                    save_dir=tmp_path,
                    train_set_size=20,
                    device='cuda:0')
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)

def test_load_ckpt_repeat(tmp_path):
    model = resnet18()
    traker = TRAKer(model=model,
                    task='image_classification',
                    save_dir=tmp_path,
                    train_set_size=20,
                    device='cuda:0')
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.load_checkpoint(ckpt, model_id=1)

@pytest.mark.cuda
def test_featurize(tmp_path):
    model = resnet18().cuda().eval()
    N = 5
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,)).cuda()
    traker = TRAKer(model=model,
                    task='image_classification',
                    save_dir=tmp_path,
                    train_set_size=20,
                    device='cuda:0')
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize(batch, num_samples=N)

@pytest.mark.cuda
def test_finalize_features(tmp_path):
    model = resnet18().cuda().eval()
    N = 5
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,)).cuda()
    traker = TRAKer(model=model,
                    task='image_classification',
                    save_dir=tmp_path,
                    train_set_size=N,
                    device='cuda:0')
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize(batch, num_samples=N)
    traker.finalize_features()

@pytest.mark.cuda
def test_score(tmp_path):
    model = resnet18().cuda().eval()
    N = 5
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,)).cuda()
    traker = TRAKer(model=model,
                    task='image_classification',
                    save_dir=tmp_path,
                    train_set_size=N,
                    device='cuda:0')
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize(batch, num_samples=N)
    traker.finalize_features()
    traker.score(batch, ckpt, model_id=0, num_samples=N)

@pytest.mark.cuda
def test_score_finalize(tmp_path):
    model = resnet18().cuda().eval()
    N = 5
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,)).cuda()
    traker = TRAKer(model=model,
                    task='image_classification',
                    save_dir=tmp_path,
                    train_set_size=N,
                    device='cuda:0')
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize(batch, num_samples=N)
    traker.finalize_features()

    traker.load_checkpoint(ckpt, model_id=0)
    traker.score(batch, ckpt, model_id=0, num_samples=N)
    traker.finalize_scores()