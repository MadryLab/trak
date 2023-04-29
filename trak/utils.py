from torch import Tensor
import torch
ch = torch


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


def vectorize_and_ignore_buffers(g, params_dict=None) -> Tensor:
    """
    gradients are given as a dict :code:`(name_w0: grad_w0, ... name_wp:
    grad_wp)` where :code:`p` is the number of weight matrices. each
    :code:`grad_wi` has shape :code:`[batch_size, ...]` this function flattens
    :code:`g` to have shape :code:`[batch_size, num_params]`.
    """
    batch_size = len(g[next(iter(g))][0])
    out = []
    if params_dict is not None:
        for b in range(batch_size):
            out.append(ch.cat([x[b].flatten() for i, x in enumerate(g.values()) if is_not_buffer(i, params_dict)]))
    else:
        for b in range(batch_size):
            out.append(ch.cat([x[b].flatten() for x in g]))
    return ch.stack(out)


def vectorize(g) -> Tensor:
    """
    gradients are given as a dict :code:`(name_w0: grad_w0, ... name_wp:
    grad_wp)` where :code:`p` is the number of weight matrices. each
    :code:`grad_wi` has shape :code:`[batch_size, ...]` this function flattens
    :code:`g` to have shape :code:`[batch_size, num_params]`.
    """
    batch_size = len(g[next(iter(g))])
    out = []
    for b in range(batch_size):
        out.append(ch.cat([x[b].flatten() for x in g.values()]))
    return ch.stack(out)
