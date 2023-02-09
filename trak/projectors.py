from abc import ABC, abstractmethod
from enum import Enum
from torch import Tensor
import torch as ch

class ProjectionType(Enum):
    normal = 'normal'
    rademacher = 'rademacher'

class AbstractProjector(ABC):
    """
    Implementations of the Projector class must implement the `project` method,
    which takes in model gradients and returns 
    """
    @abstractmethod
    def __init__(self,
                 grad_dim: int,
                 proj_dim: int,
                 seed: int,
                 proj_type: ProjectionType,
                 device) -> None:
        self.grad_dim = grad_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.proj_type = proj_type
        self.device = device

    @abstractmethod
    def project(self, grads: Tensor) -> Tensor:
        ...

class BasicProjector(AbstractProjector):
    """
    A bare-bones implementation of the projection, which is (extremely)
    inefficient in terms of both time and memory footrpint.
    """
    def __init__(self, grad_dim: int, proj_dim: int, seed: int, proj_type:
                 ProjectionType, device, dtype=ch.float16) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

        self.proj_type = proj_type
        self.generator = ch.Generator(device=self.device)
        self.generator = self.generator.manual_seed(self.seed)
        self.dtype = dtype

        self.proj_matrix = ch.ones(self.grad_dim,
                                   self.proj_dim,
                                   dtype=self.dtype,
                                   device=self.device)

        if self.proj_type == ProjectionType.normal or self.proj_type == 'normal':
            self.proj_matrix.normal_(generator=self.generator)
        elif self.proj_type == ProjectionType.rademacher or self.proj_type == 'rademacher':
            self.proj_matrix.bernoulli_(p=0.5, generator=self.generator)
            self.proj_matrix *= 2.
            self.proj_matrix -= 1.
        else:
            raise KeyError(f'Projection type {self.proj_type} not recognized.')

    
    def project(self, grads: Tensor) -> Tensor:
        return grads @ self.proj_matrix


class CudaProjector(AbstractProjector):
    """
    An implementation of the project for cuda (with compute capability >= 7.0)
    """
    def __init__(self, grad_dim: int, proj_dim: int, seed: int, proj_type:
                 ProjectionType, device, dtype=ch.float16) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

        self.dtype = dtype

        if self.dtype != ch.float16:
            raise NotImplementedError("Only float16 supported with the CudaProjector for now")

        if self.proj_type == ProjectionType.normal or self.proj_type == 'normal':
            raise NotImplementedError(f"{self.proj_type} not implemented yet for CudaProjector")
        elif self.proj_type == ProjectionType.rademacher or self.proj_type == 'rademacher':
            pass
        else:
            raise KeyError(f'Projection type {self.proj_type} not recognized.')

        import fast_jl
        self.projection_function = fast_jl.rademacher

    def project(self, grads: Tensor) -> Tensor:
        result = self.projection_function(grads, self.proj_dim, self.seed)
        return result
