import torch as ch

def parameters_to_vector(parameters):
    """
    Same as https://pytorch.org/docs/stable/generated/torch.nn.utils.parameters_to_vector.html
    but with `reshape` instead of `view` to avoid a pesky error. 
    """
    vec = []
    for param in parameters:
        vec.append(param.reshape(-1))
    return ch.cat(vec)


def get_num_params(model: ch.nn.Module):
    return parameters_to_vector(model.parameters()).numel()


def get_params_dict(model):
    return [x[0] for x in list(model.named_parameters())]


def is_not_buffer(ind, params_dict):
    name = params_dict[ind]
    if ('running_mean' in name) or ('running_var' in name) or ('num_batches_tracked' in name):
        return False
    return True


def vectorize_and_ignore_buffers(g, params_dict=None):
    """
    gradients are given as a tuple (grad_w0, grad_w1, ... grad_wp)
    where p is the number of weight matrices. each grad_wi has shape
    [batch_size, ...]
    this f-n flattens g to have shape [batch_size, num_params]
    """
    batch_size = len(g[0])
    out = []
    if params_dict is not None:
        for b in range(batch_size):
            out.append(ch.cat([x[b].flatten() for i, x in enumerate(g) if is_not_buffer(i, params_dict)]))
    else:
        for b in range(batch_size):
            out.append(ch.cat([x[b].flatten() for x in g]))
    return ch.stack(out)