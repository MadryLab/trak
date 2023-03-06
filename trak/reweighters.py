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
    """ A bare-bones implementation of the reweighting, inefficient in terms of
    both time and memory footrpint. Only useful for small-scale applications.
    """
    def __init__(self, device, dtype=ch.float16) -> None:
        super().__init__(device)
        self.dtype = dtype
    
    def reweight(self, grads: Tensor) -> Tensor:
        return grads.T @ grads

    def finalize(self, grads: Tensor, xtx: Tensor) -> None:
        return grads @ ch.linalg.inv(xtx) 


class BasicReweighter(AbstractReweighter):
    """ An implementation of Reweighter that computes the matrix product in a
    block-wise manner.
    """
    def __init__(self, device, dtype=ch.float16, CUDA_MAX_DIM_SIZE=100_000) -> None:
        super().__init__(device)
        self.dtype = dtype
        self.CUDA_MAX_DIM_SIZE = CUDA_MAX_DIM_SIZE
    
    def reweight(self, grads: Tensor) -> Tensor:
        self.proj_dim = grads.shape[1]
        result = ch.zeros(self.proj_dim, self.proj_dim).to(self.device)
        blocks = ch.split(grads, split_size_or_sections=self.CUDA_MAX_DIM_SIZE, dim=0)

        for block in blocks:
            result += block.T.to(self.device) @ block.to(self.device)

        return result

    def finalize(self, grads: Tensor, xtx: Tensor) -> None:
        """ Update matrix of (reweighted) dot products
        """
        blocks = ch.split(grads, split_size_or_sections=self.CUDA_MAX_DIM_SIZE, dim=0)
        xtx_inv = ch.linalg.inv(xtx)

        result = ch.zeros(grads.shape[0], xtx_inv.shape[1], device=self.device)
        for i, block in enumerate(blocks):
            start = i * self.CUDA_MAX_DIM_SIZE
            end = min(grads.shape[0], (i + 1) * self.CUDA_MAX_DIM_SIZE)
            result[start : end] += (block.to(self.device) @ xtx_inv)
        return result