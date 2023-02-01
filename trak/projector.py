from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable, Optional, Literal
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
                 ProjectionType, device) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

        self.generator = ch.Generator(device=self.device)
        self.generator = self.generator.manual_seed(self.seed)

        self.proj_matrix = ch.empty(self.grad_dim, self.proj_dim)

        if self.proj_type == ProjectionType.normal:
            self.proj_matrix.normal_(generator=self.generator)
        elif self.proj_type == ProjectionType.rademacher:
            self.proj_matrix.bernoulli_(p=0.5, generator=self.generator)

    
    def project(self, grads: Tensor) -> Tensor:
        return grads @ self.proj_matrix