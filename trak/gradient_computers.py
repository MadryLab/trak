from abc import ABC, abstractmethod
from typing import Iterable, Optional
from torch import Tensor
from .utils import parameters_to_vector, vectorize_and_ignore_buffers, get_params_dict
from .modelout_functions import AbstractModelOutput
import torch
ch = torch
try:
    from functorch import grad, vmap, make_functional_with_buffers
except ImportError:
    print('Cannot import `functorch`. Functional mode cannot be used.')


class AbstractGradientComputer(ABC):
    """
    Implementations of the GradientComputer class should allow for per-sample
    gradients.
    This is behavior is enabled with three methods:
    - the `load_model_params` method, well, loads model parameters. It can be as
      simple as a self.model.load_state_dict(..)
    - the `compute_per_sample_grad` method computes per-sample gradients of the
      chosen model output function with respect to the model's parameters.
    - the `compute_loss_grad` method computes the gradients of the loss function
       with respect to the model output (which should be a scalar) for every
       sample.

    The class attribute `is_functional` is used to determine what implementation
    of ModelOutput to use, i.e. whether it should use `functorch`'s functional
    models.
    """
    is_functional = True

    @abstractmethod
    def __init__(self,
                 model: torch.nn.Module,
                 modelout_fn: AbstractModelOutput,
                 grad_dim: Optional[int] = None,
                 ) -> None:
        """ Initializes attributes, nothing too interesting happening.

        Args:
            model (torch.nn.Module): model
            modelout_fn (AbstractModelOutput): model output function
            grad_dim (int, optional): Size of the gradients (number of model
                parameters). Defaults to None.
        """
        self.model = model
        self.modelout_fn = modelout_fn()
        self.grad_dim = grad_dim

    @abstractmethod
    def load_model_params(self, model) -> None:
        ...

    @abstractmethod
    def compute_per_sample_grad(self, batch: Iterable[Tensor], batch_size: int) -> Tensor:
        ...

    @abstractmethod
    def compute_loss_grad(self, batch: Iterable[Tensor], batch_size: int) -> Tensor:
        ...


class FunctionalGradientComputer(AbstractGradientComputer):
    def __init__(self,
                 model: torch.nn.Module,
                 modelout_fn: AbstractModelOutput,
                 grad_dim: int) -> None:
        super().__init__(model, modelout_fn, grad_dim)
        self.load_model_params(model)

    def load_model_params(self, model) -> None:
        """ Given a a torch.nn.Module model, inits/updates the func_model, along
        with its weights and buffers. See
        https://pytorch.org/functorch/stable/generated/functorch.make_functional_with_buffers.html#functorch-make-functional-with-buffers
        for more details on `functorch`'s functional models.

        Args:
            model (torch.nn.Module): model to load
        """
        self.func_model, self.func_weights, self.func_buffers = make_functional_with_buffers(model)
        self.params_dict = get_params_dict(model)

    def compute_per_sample_grad(self,
                                batch: Iterable[Tensor],
                                batch_size: Optional[int] = None,
                                ) -> Tensor:
        """ Uses functorch's vmap (see
        https://pytorch.org/functorch/stable/generated/functorch.vmap.html#functorch.vmap
        for more details) to vectorize the computations of per-sample gradients.

        Doesn't use `batch_size`; only added to follow the abstract method signature.

        Args:
            batch (Iterable[Tensor]): batch of data
            batch_size (int, optional): Defaults to None.

        Returns:
            Tensor: gradients of the model output function of each sample in the batch
                with respect to the model's parameters.
        """
        # taking the gradient wrt weights (second argument of get_output, hence argnums=1)
        grads_loss = grad(self.modelout_fn.get_output, has_aux=False, argnums=1)
        # map over batch dimensions (hence 0 for each batch dimension, and None for model params)
        grads = vmap(grads_loss,
                     in_dims=(None, None, None, *([0] * len(batch))),
                     randomness='different')(self.func_model, self.func_weights,
                                             self.func_buffers, *batch)
        return vectorize_and_ignore_buffers(grads, self.params_dict)

    def compute_loss_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Computes the gradient of the loss with respect to the model output
        .. math::
            \\partial \\ell / \\partial \\text{model output}

        Note: For all applications we considered, we analytically derived the
        out-to-loss gradient, thus avoiding the need to do any backward passes
        (let alone per-sample grads). If for your application this is not feasible,
        you'll need to subclass this and modify this method to have a structure
        similar to the one of `self.get_output`, i.e. something like:
        ```
        grad_out_to_loss = grad(self.model_out_to_loss_grad, ...)
        grads = vmap(grad_out_to_loss, ...)
        ...
        ```
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
                 grad_dim: int) -> None:
        super().__init__(model, modelout_fn, grad_dim)
        self.load_model_params(model)

    def load_model_params(self, model) -> Tensor:
        self.model = model
        self.model_params = list(self.model.parameters())

    def compute_per_sample_grad(self,
                                batch: Iterable[Tensor],
                                batch_size: int,
                                ) -> Tensor:
        """ Computes per-sample gradients of the model output function This
        method does not leverage vectorization (and is hence much slower than
        its equivalent in `FunctionalGradientComputer`). We recommend that
        you use this only if `functorch` is not available to you, e.g. you have
        a (very) old version of pytorch.
        """
        grads = ch.zeros(batch_size, self.grad_dim).to(batch[0].device)

        margin = self.modelout_fn.get_output(self.model, *batch)
        for ind in range(batch_size):
            grads[ind] = parameters_to_vector(ch.autograd.grad(margin[ind],
                                                               self.model_params,
                                                               retain_graph=True))
        return grads

    def compute_loss_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Computes the gradient of the loss with respect to the model output
        .. math::
            \\partial \\ell / \\partial \\text{model output}

        Note: For all applications we considered, we analytically derived the
        out-to-loss gradient, thus avoiding the need to do any backward passes
        (let alone per-sample grads). If for your application this is not feasible,
        you'll need to subclass this and modify this method to have a structure
        similar to the one of `self.get_output`, i.e. something like:
        ```
        out_to_loss = self.model_out_to_loss(...)
        for ind in range(batch_size):
            grads[ind] = torch.autograd.grad(out_to_loss[ind], ...)
        ...
        ```
        """
        return self.modelout_fn.get_out_to_loss_grad(self.model, batch)
