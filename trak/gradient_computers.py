from abc import ABC, abstractmethod
from typing import Iterable, Optional
from torch import Tensor
from .utils import vectorize
from .modelout_functions import AbstractModelOutput
import torch
ch = torch


class AbstractGradientComputer(ABC):
    """ Implementations of the GradientComputer class should allow for
    per-sample gradients.  This is behavior is enabled with three methods:

    - the :meth:`.load_model_params` method, well, loads model parameters. It can
      be as simple as a :code:`self.model.load_state_dict(..)`

    - the :meth:`.compute_per_sample_grad` method computes per-sample gradients
      of the chosen model output function with respect to the model's parameters.

    - the :meth:`.compute_loss_grad` method computes the gradients of the loss
      function with respect to the model output (which should be a scalar) for
      every sample.

    """

    @abstractmethod
    def __init__(self,
                 model: torch.nn.Module,
                 task: AbstractModelOutput,
                 grad_dim: Optional[int] = None,
                 ) -> None:
        """ Initializes attributes, nothing too interesting happening.

        Args:
            model (torch.nn.Module):
                model
            task (AbstractModelOutput):
                task (model output function)
            grad_dim (int, optional):
                Size of the gradients (number of model parameters). Defaults to
                None.

        """
        self.model = model
        self.modelout_fn = task
        self.grad_dim = grad_dim

    @abstractmethod
    def load_model_params(self, model) -> None:
        ...

    @abstractmethod
    def compute_per_sample_grad(self, batch: Iterable[Tensor]) -> Tensor:
        ...

    @abstractmethod
    def compute_loss_grad(self, batch: Iterable[Tensor], batch_size: int) -> Tensor:
        ...


class FunctionalGradientComputer(AbstractGradientComputer):
    def __init__(self,
                 model: torch.nn.Module,
                 task: AbstractModelOutput,
                 grad_dim: int) -> None:
        super().__init__(model, task, grad_dim)
        self.model = model
        self.load_model_params(model)

    def load_model_params(self, model) -> None:
        """ Given a a torch.nn.Module model, inits/updates the (functional)
        weights and buffers. See https://pytorch.org/docs/stable/func.html
        for more details on :code:`torch.func`'s functional models.

        Args:
            model (torch.nn.Module):
                model to load

        """
        self.func_weights = dict(model.named_parameters())
        self.func_buffers = dict(model.named_buffers())

    def compute_per_sample_grad(self,
                                batch: Iterable[Tensor],
                                ) -> Tensor:
        """ Uses functorch's :code:`vmap` (see
        https://pytorch.org/functorch/stable/generated/functorch.vmap.html#functorch.vmap
        for more details) to vectorize the computations of per-sample gradients.

        Doesn't use :code:`batch_size`; only added to follow the abstract method
        signature.

        Args:
            batch (Iterable[Tensor]):
                batch of data

        Returns:
            Tensor:
                gradients of the model output function of each sample in the
                batch with respect to the model's parameters.

        """
        # taking the gradient wrt weights (second argument of get_output, hence argnums=1)
        grads_loss = torch.func.grad(self.modelout_fn.get_output, has_aux=False, argnums=1)
        # map over batch dimensions (hence 0 for each batch dimension, and None for model params)
        grads = torch.func.vmap(grads_loss,
                                in_dims=(None, None, None, *([0] * len(batch))),
                                randomness='different')(self.model,
                                                        self.func_weights,
                                                        self.func_buffers,
                                                        *batch)
        return vectorize(grads)

    def compute_loss_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Computes the gradient of the loss with respect to the model output

        .. math::

            \\partial \\ell / \\partial \\text{(model output)}

        Note: For all applications we considered, we analytically derived the
        out-to-loss gradient, thus avoiding the need to do any backward passes
        (let alone per-sample grads). If for your application this is not feasible,
        you'll need to subclass this and modify this method to have a structure
        similar to the one of :meth:`FunctionalGradientComputer:.get_output`,
        i.e. something like:

        .. code-block:: python

            grad_out_to_loss = grad(self.model_out_to_loss_grad, ...)
            grads = vmap(grad_out_to_loss, ...)
            ...

        Args:
            batch (Iterable[Tensor]):
                batch of data

        """
        return self.modelout_fn.get_out_to_loss_grad(self.model,
                                                     self.func_weights,
                                                     self.func_buffers,
                                                     batch)
