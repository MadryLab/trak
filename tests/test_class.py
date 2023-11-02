from trak import TRAKer
from torchvision.models import resnet18
import logging
import pytest
import torch as ch
from trak.projectors import BasicProjector, NoOpProjector
from trak.modelout_functions import ImageClassificationModelOutput


@pytest.fixture
def cpu_proj():
    projector = BasicProjector(
        grad_dim=11689512, proj_dim=20, seed=0, proj_type="rademacher", device="cpu"
    )
    return projector


def test_class_init_cpu(tmp_path, cpu_proj):
    model = resnet18()
    TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        projector=cpu_proj,
        train_set_size=20,
        logging_level=logging.DEBUG,
        device="cpu",
    )


def test_class_init(tmp_path, cpu_proj):
    model = resnet18()
    TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        projector=cpu_proj,
        train_set_size=20,
        logging_level=logging.DEBUG,
        device="cuda:0",
    )


def test_load_ckpt(tmp_path, cpu_proj):
    model = resnet18()
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        projector=cpu_proj,
        train_set_size=20,
        logging_level=logging.DEBUG,
        device="cuda:0",
    )
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)


def test_load_ckpt_repeat(tmp_path, cpu_proj):
    model = resnet18()
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        projector=cpu_proj,
        train_set_size=20,
        logging_level=logging.DEBUG,
        device="cuda:0",
    )
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.load_checkpoint(ckpt, model_id=1)


@pytest.mark.cuda
def test_featurize(tmp_path):
    model = resnet18().cuda().eval()
    N = 32
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,)).cuda()
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        train_set_size=N,
        logging_level=logging.DEBUG,
        device="cuda:0",
    )
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize(batch, num_samples=N)


@pytest.mark.cuda
def test_max_batch_size(tmp_path):
    model = resnet18().cuda().eval()
    N = 32
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,)).cuda()
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        train_set_size=N,
        logging_level=logging.DEBUG,
        proj_max_batch_size=16,
        device="cuda:0",
    )
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize(batch, num_samples=N)


def test_class_featurize_cpu(tmp_path, cpu_proj):
    model = resnet18()
    N = 5
    batch = ch.randn(N, 3, 32, 32), ch.randint(low=0, high=10, size=(N,))
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        projector=cpu_proj,
        train_set_size=N,
        logging_level=logging.DEBUG,
        device="cpu",
    )

    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize(batch, num_samples=N)


def test_class_featurize_noop(tmp_path):
    model = resnet18()
    N = 5
    batch = ch.randn(N, 3, 32, 32), ch.randint(low=0, high=10, size=(N,))
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        projector=NoOpProjector(device="cpu"),
        train_set_size=N,
        logging_level=logging.DEBUG,
        device="cpu",
    )

    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize(batch, num_samples=N)


@pytest.mark.cuda
def test_forgot_loading_ckpt(tmp_path):
    model = resnet18().cuda().eval()
    N = 5
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,)).cuda()
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        train_set_size=N,
        logging_level=logging.DEBUG,
        device="cuda:0",
    )
    with pytest.raises(
        AssertionError,
        match="Load a checkpoint using traker.load_checkpoint before featurizing",
    ):
        traker.featurize(batch, num_samples=N)


@pytest.mark.cuda
def test_finalize_features(tmp_path):
    model = resnet18().cuda().eval()
    N = 5
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,)).cuda()
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        train_set_size=N,
        logging_level=logging.DEBUG,
        device="cuda:0",
    )
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize(batch, num_samples=N)
    traker.finalize_features()


@pytest.mark.cuda
def test_finalize_features_multiple_ftr(tmp_path):
    model = resnet18().cuda().eval()
    N = 10
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,)).cuda()
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        train_set_size=N,
        logging_level=logging.DEBUG,
        device="cuda:0",
    )
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize([x[:3] for x in batch], num_samples=3)
    traker.featurize([x[3:6] for x in batch], num_samples=3)
    traker.featurize([x[6:] for x in batch], num_samples=4)
    traker.finalize_features()


@pytest.mark.cuda
def test_finalize_features_multiple_ftr_and_id(tmp_path):
    model = resnet18().cuda().eval()
    N = 10
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,)).cuda()
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        train_set_size=N,
        logging_level=logging.DEBUG,
        device="cuda:0",
    )
    ckpt = model.state_dict()
    for model_id in range(2):
        traker.load_checkpoint(ckpt, model_id=model_id)
        traker.featurize([x[:3] for x in batch], num_samples=3)
        traker.featurize([x[3:6] for x in batch], num_samples=3)
        traker.featurize([x[6:] for x in batch], num_samples=4)
    traker.finalize_features()


@pytest.mark.cuda
def test_score(tmp_path):
    model = resnet18().cuda().eval()
    N = 5
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,)).cuda()
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        train_set_size=N,
        logging_level=logging.DEBUG,
        device="cuda:0",
    )
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize(batch, num_samples=N)
    traker.finalize_features()
    traker.start_scoring_checkpoint("test_experiment", ckpt, 0, num_targets=N)
    traker.score(batch, num_samples=N)


@pytest.mark.cuda
def test_score_finalize(tmp_path):
    model = resnet18().cuda().eval()
    N = 5
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,)).cuda()
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        train_set_size=N,
        logging_level=logging.DEBUG,
        device="cuda:0",
    )
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize(batch, num_samples=N)
    traker.finalize_features()

    traker.start_scoring_checkpoint("test_experiment", ckpt, 0, num_targets=N)
    traker.score(batch, num_samples=N)
    traker.finalize_scores(exp_name="test_experiment")


@pytest.mark.cuda
def test_score_finalize_some_model_ids(tmp_path):
    model = resnet18().cuda().eval()
    N = 5
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,)).cuda()
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        train_set_size=N,
        logging_level=logging.DEBUG,
        device="cuda:0",
    )
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize(batch, num_samples=N)

    traker.load_checkpoint(ckpt, model_id=1)
    traker.featurize(batch, num_samples=N)
    traker.finalize_features()

    traker.start_scoring_checkpoint("test_experiment", ckpt, 0, num_targets=N)
    traker.score(batch, num_samples=N)
    traker.finalize_scores(exp_name="test_experiment", model_ids=[0])


@pytest.mark.cuda
def test_score_finalize_split(tmp_path):
    model = resnet18().cuda().eval()
    N = 5
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,)).cuda()
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        train_set_size=N,
        logging_level=logging.DEBUG,
        device="cuda:0",
    )
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize(batch, num_samples=N)

    traker.load_checkpoint(ckpt, model_id=1)
    traker.featurize(batch, num_samples=N)
    traker.finalize_features()

    traker.start_scoring_checkpoint("test_experiment", ckpt, 0, num_targets=N)
    traker.score(batch, num_samples=N)

    traker.start_scoring_checkpoint("test_experiment", ckpt, 1, num_targets=N)
    traker.score(batch, num_samples=N)

    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        train_set_size=N,
        logging_level=logging.DEBUG,
        device="cuda:0",
    )
    traker.finalize_scores(exp_name="test_experiment")


@pytest.mark.cuda
def test_score_finalize_full_precision(tmp_path):
    model = resnet18().cuda().eval()
    N = 5
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,)).cuda()
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        train_set_size=N,
        logging_level=logging.DEBUG,
        device="cuda:0",
        use_half_precision=False,
    )
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize(batch, num_samples=N)
    traker.finalize_features()

    traker.start_scoring_checkpoint("test_experiment", ckpt, 0, num_targets=N)
    traker.score(batch, num_samples=N)
    traker.finalize_scores(exp_name="test_experiment")


def test_custom_model_output(tmp_path, cpu_proj):
    model = resnet18()
    TRAKer(
        model=model,
        task=ImageClassificationModelOutput(),
        save_dir=tmp_path,
        projector=cpu_proj,
        train_set_size=20,
        logging_level=logging.DEBUG,
        device="cpu",
    )


def test_grad_wrt_last_layer(tmp_path):
    model = resnet18().eval()
    N = 5
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,))
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        train_set_size=N,
        logging_level=logging.DEBUG,
        device="cpu",
        use_half_precision=False,
        grad_wrt=["fc.weight", "fc.bias"],
    )
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize(batch, num_samples=N)
    traker.finalize_features()

    traker.start_scoring_checkpoint("test_experiment", ckpt, 0, num_targets=N)
    traker.score(batch, num_samples=N)
    traker.finalize_scores(exp_name="test_experiment")


@pytest.mark.cuda
def test_grad_wrt_last_layer_cuda(tmp_path):
    model = resnet18().cuda().eval()
    N = 5
    batch = ch.randn(N, 3, 32, 32).cuda(), ch.randint(low=0, high=10, size=(N,)).cuda()
    traker = TRAKer(
        model=model,
        task="image_classification",
        save_dir=tmp_path,
        train_set_size=N,
        logging_level=logging.DEBUG,
        device="cuda:0",
        grad_wrt=["fc.weight", "fc.bias"],
    )
    ckpt = model.state_dict()
    traker.load_checkpoint(ckpt, model_id=0)
    traker.featurize(batch, num_samples=N)
    traker.finalize_features()

    traker.start_scoring_checkpoint("test_experiment", ckpt, 0, num_targets=N)
    traker.score(batch, num_samples=N)
    traker.finalize_scores(exp_name="test_experiment")
