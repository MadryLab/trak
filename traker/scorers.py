from abc import ABC, abstractmethod
from torch import Tensor
import torch as ch
from .utils import parameters_to_vector, vectorize_and_ignore_buffers
try:
    from functorch import make_functional_with_buffers, vmap, grad
except:
    print('Cannot import `functorch`. Functional mode cannot be used.')

class AbstractScorer(ABC):
    """
    Implementations of the Scorer class must implement the `score` method
    """
    @abstractmethod
    def __init__(self,
                 device,
                 projector,
                 grad_dtype=ch.float16) -> None:
        self.device = device
        self.projector = projector
        self.grad_dtype = grad_dtype

    @abstractmethod
    def score(self, grads: Tensor, model_id: int) -> Tensor:
        ...

class FunctionalScorer(AbstractScorer):
    def __init__(self, device, projector, grad_dtype, **kwargs) -> None:
        super().__init__(device, projector, grad_dtype)
    
    def score(self, features, out_fn, model, model_id, batch) -> Tensor:
        _, weights, buffers = make_functional_with_buffers(model)
        grads_loss = grad(out_fn, has_aux=False)
        # map over batch dimension
        grads = vmap(grads_loss,
                     in_dims=(None, None, *([0] * len(batch))),
                     randomness='different')(weights, buffers, *batch)
        grads = self.projector.project(vectorize_and_ignore_buffers(grads).to(self.grad_dtype),
                                       model_id=model_id)

        return grads.detach().clone() @ features[model_id].T

class IterScorer(AbstractScorer):
    def __init__(self, device, projector, grad_dtype=ch.float16,
                 grad_dim=1, grad_wrt=None) -> None:
        super().__init__(device, projector, grad_dtype)
        self.grad_dim = grad_dim
        self.grad_wrt = grad_wrt
    
    def score(self, features, out_fn, model, model_id, batch) -> Tensor:
        # assuming batch is an iterable of torch Tensors,
        # each of shape [batch_size, ...]
        batch_size = batch[0].size(0)

        grads = ch.zeros(batch_size, self.grad_dim).to(self.device)
        margin = out_fn(model, *batch)

        grad_wrt = self.grad_wrt if self.grad_wrt is not None else list(model.parameters())
        for ind in range(batch_size):
            grads[ind] = parameters_to_vector(ch.autograd.grad(margin[ind],
                                                               grad_wrt,
                                                               retain_graph=True))

        grads = self.projector.project(grads.to(self.grad_dtype), model_id=model_id)
        return grads.detach().clone() @ features[model_id].T