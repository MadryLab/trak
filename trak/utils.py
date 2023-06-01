from torch import Tensor
import tempfile
import torch
ch = torch


def test_install(use_fast_jl: bool = True):
    try:
        from trak import TRAKer
    except ImportError:
        raise ImportError('TRAK is not installed! Please install it using `pip install traker`')

    data = (ch.randn(20, 256), ch.randint(high=2, size=(20,)))
    model = ch.nn.Linear(256, 2, bias=False)

    if use_fast_jl:
        with tempfile.TemporaryDirectory() as tmpdirname:
            data = [x.cuda() for x in data]
            model = model.cuda()
            traker = TRAKer(model=model,
                            task='image_classification',
                            proj_dim=512,
                            save_dir=tmpdirname,
                            train_set_size=20,
                            logging_level=100)
            traker.load_checkpoint(model.state_dict(), model_id=0)
            traker.featurize(data, num_samples=20)
            print('TRAK and fast_jl are installed correctly!')
    else:
        from trak.projectors import NoOpProjector
        with tempfile.TemporaryDirectory() as tmpdirname:
            traker = TRAKer(model=model,
                            task='image_classification',
                            train_set_size=20,
                            proj_dim=512,
                            save_dir=tmpdirname,
                            projector=NoOpProjector(),
                            device='cpu',
                            logging_level=100)
            traker.load_checkpoint(model.state_dict(), model_id=0)
            traker.featurize(data, num_samples=20)
            print('TRAK is installed correctly!')


def parameters_to_vector(parameters) -> Tensor:
    """
    Same as https://pytorch.org/docs/stable/generated/torch.nn.utils.parameters_to_vector.html
    but with :code:`reshape` instead of :code:`view` to avoid a pesky error.
    """
    vec = []
    for param in parameters:
        vec.append(param.reshape(-1))
    return ch.cat(vec)


def get_num_params(model: torch.nn.Module) -> int:
    return parameters_to_vector(model.parameters()).numel()


def is_not_buffer(ind, params_dict) -> bool:
    name = params_dict[ind]
    if ('running_mean' in name) or ('running_var' in name) or ('num_batches_tracked' in name):
        return False
    return True


def vectorize(g, arr) -> Tensor:
    """
    records result into arr

    gradients are given as a dict :code:`(name_w0: grad_w0, ... name_wp:
    grad_wp)` where :code:`p` is the number of weight matrices. each
    :code:`grad_wi` has shape :code:`[batch_size, ...]` this function flattens
    :code:`g` to have shape :code:`[batch_size, num_params]`.
    """
    pointer = 0
    for param in g.values():
        num_param = param[0].numel()
        arr[:, pointer:pointer + num_param] = param.flatten(start_dim=1).data
        pointer += num_param
