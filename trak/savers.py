from abc import ABC, abstractmethod
from torch import Tensor
import torch as ch
from typing import Optional

class AbstractSaver(ABC):
    """
    Implementations of Saver class must implement getters and setters
    for `grads` and `loss_grads`, as well as the methods `finalize` and
    `load`.
    """
    @abstractmethod
    def __init__(self,
                 save_dir,
                 device) -> None:
        self.device = device
        self.save_dir = save_dir
        self.model_ids = set()

    @abstractmethod
    def grad_set(self, grads: Tensor) -> None:
        ...

    @abstractmethod
    def grad_get(self, inds: Optional[Tensor]) -> None:
        ...

    @abstractmethod
    def loss_set(self, loss_grads: Tensor) -> None:
        ...

    @abstractmethod
    def loss_get(self, inds: Optional[Tensor]) -> None:
        ...

    @abstractmethod
    def finalize(self) -> None:
        ...

    @abstractmethod
    def load(self, *args) -> None:
        ...


class KeepInRAMSaver(AbstractSaver):
    """ A basic "saver" that does not serialize anything and
    instead keeps all tensors in RAM.
    """
    def __init__(self, device, save_dir, grads_shape) -> None:
        super().__init__(save_dir, device)
        self.grads_shape = grads_shape
        self.loss_shape = [grads_shape[0], 1]
        self.grads = {0: ch.zeros(self.grads_shape, device=self.device)}
        self.loss_grads = {0: ch.zeros(self.loss_shape, device=self.device)}
        self.model_ids.add(0)
    
    def grad_set(self, grads: Tensor, inds: Tensor, model_id=0) -> None:
        if self.grads.get(model_id) is None:
            self.model_ids.add(model_id)
            self.grads[model_id] = ch.zeros(self.grads_shape, device=self.device)
        self.grads[model_id][inds] = grads

    def grad_get(self, inds: Optional[Tensor]=None, model_id=0) -> Tensor:
        if inds == None:
            return self.grads[model_id]
        else:
            return self.grads[model_id][inds]

    def loss_set(self, loss_grads: Tensor, inds: Tensor, model_id=0) -> None:
        if self.loss_grads.get(model_id) is None:
            self.model_ids.add(model_id)
            self.loss_grads[model_id] = ch.zeros(self.loss_shape, device=self.device)
        self.loss_grads[model_id][inds] = loss_grads.unsqueeze(-1)

    def loss_get(self, inds: Optional[Tensor]=None, model_id=0) -> Tensor:
        if inds == None:
            return self.loss_grads[model_id]
        else:
            return self.loss_grads[model_id][inds]

    def finalize(self) -> None:
        pass

    def load(self, grads, loss_grads) -> None:
        self.grads = grads
        self.loss_grads = loss_grads