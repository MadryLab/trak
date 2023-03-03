from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union
from torch import Tensor
import torch
ch = torch
from .utils import parameters_to_vector, vectorize_and_ignore_buffers, get_params_dict
from .modelout_functions import AbstractModelOutput
try:
    from functorch import grad, vmap, make_functional_with_buffers
except ImportError:
    print('Cannot import `functorch`. Functional mode cannot be used.')


class AbstractGradientComputer(ABC):
    """
    Implementations of the GradientComputer class must implement
    the `compute_per_sample_grad` method.
    """
    is_functional = True
    @abstractmethod
    def __init__(self,
                 model,
                 modelout_fn,
                 device,
                 grad_dim=None,
                 ) -> None:
        self.model = model
        self.modelout_fn = modelout_fn()
        self.device = device
        self.grad_dim = grad_dim

    @abstractmethod
    def load_model_params(self, model, func_model, func_weights, func_buffers) -> Tensor:
        ...

    @abstractmethod
    def compute_per_sample_grad(self, batch: Iterable[Tensor], batch_size: int) -> Tensor:
        ...

    @abstractmethod
    def compute_loss_grad(self, batch: Iterable[Tensor], batch_size: int) -> Tensor:
        ...
    


class FunctionalGradientComputer(AbstractGradientComputer):
    def __init__(self,
                 model,
                 modelout_fn: AbstractModelOutput,
                 device: Union[torch.device, str],
                 grad_dim: int) -> None:
        super().__init__(model, modelout_fn, device, grad_dim)
        self.load_model_params(model)
    
    def load_model_params(self, model) -> Tensor:
        self.func_model, self.func_weights, self.func_buffers = make_functional_with_buffers(model)
        self.params_dict = get_params_dict(model)
    
    def compute_per_sample_grad(self,
                                batch: Iterable[Tensor],
                                batch_size: Optional[int]=None,
                                ) -> Tensor:
        # taking the gradient wrt weights (second argument of get_output, hence argnums=1)
        grads_loss = grad(self.modelout_fn.get_output, has_aux=False, argnums=1)
        # map over batch dimensions (hence 0 for each batch dimension, and None for model params)
        grads = vmap(grads_loss,
                     in_dims=(None, None, None, *([0] * len(batch))),
                     randomness='different')(self.func_model, self.func_weights,
                                             self.func_buffers, *batch)
        return vectorize_and_ignore_buffers(grads, self.params_dict)
    
    def compute_loss_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Computes
        .. math::
            \partial \ell / \partial \text{margin}
        
        """
        return self.modelout_fn.get_out_to_loss_grad(self.func_model,
                                                     self.func_weights,
                                                     self.func_buffers,
                                                     batch)


class IterativeGradientComputer(AbstractGradientComputer):
    is_functional = False 
    def __init__(self,
                 model,
                 modelout_fn: AbstractModelOutput,
                 device: Union[torch.device, str],
                 grad_dim: int) -> None:
        super().__init__(model, modelout_fn, device, grad_dim)
        self.load_model_params(model)
    
    def load_model_params(self, model) -> Tensor:
        self.model = model
        self.model_params = list(self.model.parameters())

    def compute_per_sample_grad(self,
                                batch: Iterable[Tensor],
                                batch_size: int,
                                ) -> Tensor:
        """Computes per-sample gradients of the model output function
        This method does not leverage vectorization (and is hence much slower than
        `_featurize_vmap`).
        """
        grads = ch.zeros(batch_size, self.grad_dim).to(self.device)

        margin = self.modelout_fn.get_output(self.model, *batch)
        for ind in range(batch_size):
            grads[ind] = parameters_to_vector(ch.autograd.grad(margin[ind],
                                                               self.model_params,
                                                               retain_graph=True))
        return grads
    
    def compute_loss_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Computes
        .. math::
            \partial \ell / \partial \text{margin}
        
        """
        return self.modelout_fn.get_out_to_loss_grad(self.model, batch)