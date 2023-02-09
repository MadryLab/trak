from abc import ABC, abstractmethod
from torch import Tensor
import torch as ch
from typing import Optional

class AbstractReweighter(ABC):
    """
    Implementations of the Reweighter class must implement the `reweight`
    and `finalize` methods.
    """
    @abstractmethod
    def __init__(self,
                 device) -> None:
        self.device = device

    @abstractmethod
    def reweight(self, grads: Tensor) -> Tensor:
        ...

    @abstractmethod
    def finalize(self, grads: Tensor, xtx: Tensor) -> Optional[Tensor]:
        ...


class BasicSingleBlockReweighter(AbstractReweighter):
    """ A bare-bones implementation of the reweighting, inefficient
    in terms of both time and memory footrpint.  Only useful for
    small-scale applications.
    """
    def __init__(self, device, dtype=ch.float16) -> None:
        super().__init__(device)
        self.dtype = dtype
    
    def reweight(self, grads: Tensor) -> Tensor:
        return grads.T @ grads

    def finalize(self, grads: Tensor, xtx: Tensor) -> Tensor:
        return grads @ ch.linalg.inv(xtx)