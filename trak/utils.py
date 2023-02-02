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