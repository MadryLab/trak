from abc import ABC, abstractmethod
from torch import Tensor
import torch as ch


class AbstractScoreComputer(ABC):
    """
    The :code:`ScoreComputer` class
    Implementations of the ScoreComputer class must implement three methods:
    - :code:`get_xtx`
    - :code:`get_x_xtx_inv`
    - :code:`get_scores`
    """
    @abstractmethod
    def __init__(self, device) -> None:
        self.device = device

    @abstractmethod
    def get_xtx(self, grads: Tensor) -> Tensor:
        ...

    @abstractmethod
    def get_x_xtx_inv(self, grads: Tensor, xtx: Tensor) -> Tensor:
        ...

    @abstractmethod
    def get_scores(self, features: Tensor, target_grads: Tensor) -> Tensor:
        ...


class BasicSingleBlockScoreComputer(AbstractScoreComputer):
    """ A bare-bones implementation of :code:`ScoreComputer` that will likely
    OOM for almost all applications. Here for testing purposes only. Unless you
    have a good reason not to, you should use :func:`BasicScoreComputer`
    instead.
    """
    def __init__(self, device) -> None:
        super().__init__(device)

    def get_xtx(self, grads: Tensor) -> Tensor:
        return grads.T @ grads

    def get_x_xtx_inv(self, grads: Tensor, xtx: Tensor) -> Tensor:
        return grads @ ch.linalg.inv(xtx)

    def get_scores(self, features: Tensor, target_grads: Tensor) -> Tensor:
        return features @ target_grads.T


class BasicScoreComputer(AbstractScoreComputer):
    """ An implementation of :code:`ScoreComputer` that computes matmuls in a
    block-wise manner.
    """
    def __init__(self, device, CUDA_MAX_DIM_SIZE: int = 100_000) -> None:
        """
        Args:
            device (Union[str, torch.device]): torch device to do matmuls on
            CUDA_MAX_DIM_SIZE (int, optional): Size of block for block-wise
            matmuls. Defaults to 100_000.
        """
        super().__init__(device)
        self.CUDA_MAX_DIM_SIZE = CUDA_MAX_DIM_SIZE

    def get_xtx(self, grads: Tensor) -> Tensor:
        self.proj_dim = grads.shape[1]
        result = ch.zeros(self.proj_dim, self.proj_dim).to(self.device)
        blocks = ch.split(grads, split_size_or_sections=self.CUDA_MAX_DIM_SIZE, dim=0)

        for block in blocks:
            result += block.T.to(self.device) @ block.to(self.device)

        return result

    def get_x_xtx_inv(self, grads: Tensor, xtx: Tensor) -> Tensor:
        blocks = ch.split(grads, split_size_or_sections=self.CUDA_MAX_DIM_SIZE, dim=0)
        xtx_inv = ch.linalg.inv(xtx)

        result = ch.empty(grads.shape[0], xtx_inv.shape[1], device=self.device)
        for i, block in enumerate(blocks):
            start = i * self.CUDA_MAX_DIM_SIZE
            end = min(grads.shape[0], (i + 1) * self.CUDA_MAX_DIM_SIZE)
            result[start: end] = (block.to(self.device) @ xtx_inv)
        return result

    def get_scores(self, features: Tensor, target_grads: Tensor) -> Tensor:
        train_dim = features.shape[0]
        target_dim = target_grads.shape[0]

        if target_dim < self.CUDA_MAX_DIM_SIZE:
            return features @ target_grads.T

        result = ch.empty(train_dim, target_dim).to(self.device)
        blocks = ch.split(target_grads, split_size_or_sections=self.CUDA_MAX_DIM_SIZE, dim=0)

        for i, block in enumerate(blocks):
            start = i * self.CUDA_MAX_DIM_SIZE
            end = min(target_grads.shape[0], (i + 1) * self.CUDA_MAX_DIM_SIZE)
            result[:, start: end] = features @ block.T

        return result
