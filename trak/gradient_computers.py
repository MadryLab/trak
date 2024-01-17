"""
Computing features for the TRAK algorithm involves computing (and projecting)
per-sample gradients. This module contains classes that compute these
per-sample gradients. The :code:`AbstractFeatureComputer` class defines the
interface for such gradient computers. Then, we provide two implementations:
- :class:`FunctionalFeatureComputer`: A fast implementation that uses
  :code:`torch.func` to vectorize the computation of per-sample gradients, and
  thus fully levereage parallelism.
- :class:`IterativeFeatureComputer`: A more naive implementation that only uses
  native pytorch operations (i.e. no :code:`torch.func`), and computes per-sample
  gradients in a for-loop. This is often much slower than the functional
  version, but it is useful if you cannot use :code:`torch.func`, e.g., if you
  have an old version of pytorch that does not support it, or if your application
  is not supported by :code:`torch.func`.

"""
from abc import ABC, abstractmethod
from typing import Iterable, Optional
from torch import Tensor
from .utils import get_num_params, parameters_to_vector
from .modelout_functions import AbstractModelOutput
import logging
import torch

ch = torch


class AbstractGradientComputer(ABC):
    """Implementations of the GradientComputer class should allow for
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
    def __init__(
        self,
        model: torch.nn.Module,
        task: AbstractModelOutput,
        grad_dim: Optional[int] = None,
        dtype: Optional[torch.dtype] = torch.float16,
        device: Optional[torch.device] = "cuda",
    ) -> None:
        """Initializes attributes, nothing too interesting happening.

        Args:
            model (torch.nn.Module):
                model
            task (AbstractModelOutput):
                task (model output function)
            grad_dim (int, optional):
                Size of the gradients (number of model parameters). Defaults to
                None.
            dtype (torch.dtype, optional):
                Torch dtype of the gradients. Defaults to torch.float16.
            device (torch.device, optional):
                Torch device where gradients will be stored. Defaults to 'cuda'.

        """
        self.model = model
        self.modelout_fn = task
        self.grad_dim = grad_dim
        self.dtype = dtype
        self.device = device

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
    def __init__(
        self,
        model: torch.nn.Module,
        task: AbstractModelOutput,
        grad_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        grad_wrt: Optional[Iterable[str]] = None,
    ) -> None:
        """Initializes attributes, and loads model parameters.

        Args:
            grad_wrt (list[str], optional):
                A list of parameter names for which to keep gradients.  If None,
                gradients are taken with respect to all model parameters.
                Defaults to None.
        """
        super().__init__(model, task, grad_dim, dtype, device)
        self.model = model
        self.num_params = get_num_params(self.model)
        self.load_model_params(model)
        self.grad_wrt = grad_wrt
        self.logger = logging.getLogger("GradientComputer")

    def load_model_params(self, model) -> None:
        """Given a a torch.nn.Module model, inits/updates the (functional)
        weights and buffers. See https://pytorch.org/docs/stable/func.html
        for more details on :code:`torch.func`'s functional models.

        Args:
            model (torch.nn.Module):
                model to load

        """
        self.func_weights = dict(model.named_parameters())
        self.func_buffers = dict(model.named_buffers())

    def compute_per_sample_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Uses functorch's :code:`vmap` (see
        https://pytorch.org/functorch/stable/generated/functorch.vmap.html#functorch.vmap
        for more details) to vectorize the computations of per-sample gradients.

        Doesn't use :code:`batch_size`; only added to follow the abstract method
        signature.

        Args:
            batch (Iterable[Tensor]):
                batch of data

        Returns:
            dict[Tensor]:
                A dictionary where each key is a parameter name and the value is
                the gradient tensor for that parameter.

        """
        # taking the gradient wrt weights (second argument of get_output, hence argnums=1)
        grads_loss = torch.func.grad(
            self.modelout_fn.get_output, has_aux=False, argnums=1
        )

        # map over batch dimensions (hence 0 for each batch dimension, and None for model params)
        grads = torch.func.vmap(
            grads_loss,
            in_dims=(None, None, None, *([0] * len(batch))),
            randomness="different",
        )(self.model, self.func_weights, self.func_buffers, *batch)

        if self.grad_wrt is not None:
            for param_name in list(grads.keys()):
                if param_name not in self.grad_wrt:
                    del grads[param_name]
        return grads

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

        Returns:
            Tensor:
                The gradient of the loss with respect to the model output.
        """
        return self.modelout_fn.get_out_to_loss_grad(
            self.model, self.func_weights, self.func_buffers, batch
        )


class IterativeGradientComputer(AbstractGradientComputer):
    def __init__(
        self,
        model,
        task: AbstractModelOutput,
        grad_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        grad_wrt: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(model, task, grad_dim, dtype, device)
        self.load_model_params(model)
        self.grad_wrt = grad_wrt
        self.logger = logging.getLogger("GradientComputer")
        if self.grad_wrt is not None:
            self.logger.warning(
                "IterativeGradientComputer: ignoring grad_wrt argument."
            )

    def load_model_params(self, model) -> Tensor:
        self.model = model
        self.model_params = list(self.model.parameters())

    def compute_per_sample_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Computes per-sample gradients of the model output function This
        method does not leverage vectorization (and is hence much slower than
        its equivalent in :class:`.FunctionalGradientComputer`). We recommend
        that you use this only if :code:`torch.func` is not available to you,
        e.g. if you have a (very) old version of pytorch.
        Args:
            batch (Iterable[Tensor]):
                batch of data
        Returns:
            Tensor:
                gradients of the model output function of each sample in the
                batch with respect to the model's parameters.
        """
        batch_size = batch[0].shape[0]
        grads = ch.zeros(batch_size, self.grad_dim).to(batch[0].device)

        margin = self.modelout_fn.get_output(self.model, None, None, *batch)
        for ind in range(batch_size):
            grads[ind] = parameters_to_vector(
                ch.autograd.grad(margin[ind], self.model_params, retain_graph=True)
            )
        return grads

    def compute_loss_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Computes the gradient of the loss with respect to the model output
        .. math::
            \\partial \\ell / \\partial \\text{(model output)}
        Note: For all applications we considered, we analytically derived the
        out-to-loss gradient, thus avoiding the need to do any backward passes
        (let alone per-sample grads). If for your application this is not feasible,
        you'll need to subclass this and modify this method to have a structure
        similar to the one of :meth:`.IterativeGradientComputer.get_output`,
        i.e. something like:
        .. code-block:: python
            out_to_loss = self.model_out_to_loss(...)
            for ind in range(batch_size):
                grads[ind] = torch.autograd.grad(out_to_loss[ind], ...)
            ...
        Args:
            batch (Iterable[Tensor]):
                batch of data

        Returns:
            Tensor:
                The gradient of the loss with respect to the model output.
        """
        return self.modelout_fn.get_out_to_loss_grad(self.model, None, None, batch)
