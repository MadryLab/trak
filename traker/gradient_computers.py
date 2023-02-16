from abc import ABC, abstractmethod
from typing import Iterable, Optional
from torch import Tensor
import torch as ch
from .utils import parameters_to_vector, vectorize_and_ignore_buffers
try:
    from functorch import grad, vmap
except ImportError:
    print('Cannot import `functorch`. Functional mode cannot be used.')


class AbstractGradientComputer(ABC):
    """
    Implementations of the GradientComputer class must implement
    the `compute_per_sample_grad` method.
    """
    @abstractmethod
    def __init__(self,
                 device) -> None:
        self.device = device
        self.model_params = {}

    @abstractmethod
    def compute_per_sample_grad(self, out_fn, model_params,
                                batch: Iterable[Tensor],
                                grad_wrt: Optional[Tensor],
                                model_id: int) -> Tensor:
        ...


class FunctionalGradientComputer(AbstractGradientComputer):
    def __init__(self, func_model, device, params_dict, model_id: int=0) -> None:
        super().__init__(device)
        self.params_dict = params_dict
        self.func_models = {}
        self.register_model(func_model, model_id)
    
    def register_model(self, func_model, model_id) -> None:
        self.func_models[model_id] = func_model

    def compute_per_sample_grad(self, out_fn, 
                                batch: Iterable[Tensor],
                                grad_wrt: Optional[Tensor],
                                model_id: int) -> Tensor:
        weights, buffers = self.model_params[model_id]
        grads_loss = grad(out_fn, has_aux=False)
        # map over batch dimension
        grads = vmap(grads_loss,
                     in_dims=(None, None, *([0] * len(batch))),
                     randomness='different')(weights, buffers, *batch)
        return vectorize_and_ignore_buffers(grads, self.params_dict)
    
    def compute_loss_grad(self, loss_fn, batch, model_id: int) -> Tensor:
        """Computes
        .. math::
            \partial \ell / \partial \text{margin}
        
        """
        weights, buffers = self.model_params[model_id]
        return loss_fn(weights, buffers, *batch)


class IterativeGradientComputer(AbstractGradientComputer):
    def __init__(self, model, device, grad_dim: Tensor, model_id: int=0) -> None:
        super().__init__(device)
        self.models = {}
        self.register_model(model, model_id)
        self.grad_dim = grad_dim
    
    def register_model(self, model, model_id) -> None:
        self.models[model_id] = model

    def compute_per_sample_grad(self, out_fn,
                                batch: Iterable[Tensor],
                                grad_wrt: Optional[Tensor],
                                model_id: int) -> Tensor:
        """Computes per-sample gradients of the model output function
        This method does not leverage vectorization (and is hence much slower than
        `_featurize_vmap`).
        """
        # assuming batch is an iterable of torch Tensors, each of
        # shape [batch_size, ...]
        batch_size = batch[0].size(0)
        if grad_wrt is None:
            grad_wrt = self.model_params[model_id]

        grads = ch.zeros(batch_size, self.grad_dim).to(self.device)
        margin = out_fn(self.models[model_id], *batch)
        for ind in range(batch_size):
            grads[ind] = parameters_to_vector(ch.autograd.grad(margin[ind],
                                                               grad_wrt,
                                                               retain_graph=True))
        return grads
    
    def compute_loss_grad(self, loss_fn, batch, model_id: int) -> Tensor:
        """Computes
        .. math::
            \partial \ell / \partial \text{margin}
        
        """
        return loss_fn(self.models[model_id], *batch)