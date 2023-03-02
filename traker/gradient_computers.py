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
                 modelout_fn,
                 device) -> None:
        self.device = device
        self.modelout_fn = modelout_fn
        self.model_params = {}

    @abstractmethod
    def compute_per_sample_grad(self, model_params,
                                batch: Iterable[Tensor]) -> Tensor:
        ...


class FunctionalGradientComputer(AbstractGradientComputer):
    def __init__(self, func_model, modelout_fn, device, params_dict) -> None:
        super().__init__(modelout_fn, device)
        self.params_dict = params_dict
        self.modelout_fn = modelout_fn()
    
    def compute_per_sample_grad(self,
                                func_model,
                                weights,
                                buffers,
                                batch: Iterable[Tensor],
                                ) -> Tensor:
        # taking the gradient wrt weights (second argument of get_output, hence argnums=1)
        grads_loss = grad(self.modelout_fn.get_output, has_aux=False, argnums=1)
        # map over batch dimensions (hence 0 for each batch dimension, and None for model params)
        grads = vmap(grads_loss,
                     in_dims=(None, None, None, *([0] * len(batch))),
                     randomness='different')(func_model, weights, buffers, *batch)
        return vectorize_and_ignore_buffers(grads, self.params_dict)
    
    def compute_loss_grad(self, func_model, weights, buffers, batch) -> Tensor:
        """Computes
        .. math::
            \partial \ell / \partial \text{margin}
        
        """
        return self.modelout_fn.get_out_to_loss_grad(func_model, weights, buffers, batch)


class IterativeGradientComputer(AbstractGradientComputer):
    def __init__(self, model, modelout_fn, device, grad_dim: Tensor) -> None:
        super().__init__(modelout_fn, device)
        self.grad_dim = grad_dim
        self.modelout_fn = modelout_fn()
    
    def compute_per_sample_grad(self,
                                model,
                                batch: Iterable[Tensor],
                                batch_size: int,
                                ) -> Tensor:
        """Computes per-sample gradients of the model output function
        This method does not leverage vectorization (and is hence much slower than
        `_featurize_vmap`).
        """
        model_params = list(model.parameters())
        grads = ch.zeros(batch_size, self.grad_dim).to(self.device)

        margin = self.modelout_fn.get_output(model, *batch)
        for ind in range(batch_size):
            grads[ind] = parameters_to_vector(ch.autograd.grad(margin[ind],
                                                               model_params,
                                                               retain_graph=True))
        return grads
    
    def compute_loss_grad(self, model, batch: Iterable[Tensor]) -> Tensor:
        """Computes
        .. math::
            \partial \ell / \partial \text{margin}
        
        """
        return self.modelout_fn.get_out_to_loss_grad(model, batch)