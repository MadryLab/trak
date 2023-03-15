from abc import ABC, abstractmethod
from typing import Union
from enum import Enum
from torch import Tensor
import math
import torch
ch = torch


class ProjectionType(str, Enum):
    normal: str = 'normal'
    rademacher: str = 'rademacher'


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
                 proj_type: Union[str, ProjectionType],
                 device: Union[str, torch.device]) -> None:
        """ Initializes hyperparameters for the projection.

        Args:
            grad_dim (int): number of parameters in the model (dimension of the
                gradient vectors)
            proj_dim (int): dimension after the projection
            seed (int): random seed for the generation of the sketching
                (projection) matrix
            proj_type (Union[str, ProjectionType]): the random projection
                (JL transform) guearantees that distances will be approximately
                preserved for a variety of choices of the random matrix (see
                e.g. https://arxiv.org/abs/1411.2404). Here, we provide an
                implementation for matrices with iid Gaussian entries and iid
                Rademacher entries.
            device (Union[str, torch.device]): CUDA device to use
        """
        self.grad_dim = grad_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.proj_type = proj_type
        self.device = device

    @abstractmethod
    def project(self, grads: Tensor, model_id: int) -> Tensor:
        """ Performs the random projection. Model ID is included
        so that we generate different projection matrices for every
        model ID.

        Args:
            grads (Tensor): a batch of gradients to be projected
            model_id (int): a unique ID for a checkpoint

        Returns:
            Tensor: the projected gradients
        """
        ...


class BasicSingleBlockProjector(AbstractProjector):
    """
    A bare-bones, inefficient implementation of the projection, which simply
    calls torch's matmul for the projection step.

    Note: for most model sizes (e.g. even for ResNet18), and small projection
    dimensions (e.g. anything > 100) this method will OOM on an A100.

    Unless you have a good reason to use this class (I cannot think of one, I
    added this only for testing purposes), use instead the CudaProjector or
    BasicProjector.
    """
    def __init__(self, grad_dim: int, proj_dim: int, seed: int, proj_type:
                 ProjectionType, device, dtype=ch.float16, model_id=0) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

        self.model_id = model_id
        self.proj_type = proj_type
        self.generator = ch.Generator(device=self.device)
        self.generator = self.generator.manual_seed(self.seed + int(1e4) * self.model_id)
        self.dtype = dtype

        self.proj_matrix = ch.ones(self.grad_dim,
                                   self.proj_dim,
                                   dtype=self.dtype,
                                   device=self.device)

        self.generate_sketch_matrix()  # updates self.proj_matrix

    def generate_sketch_matrix(self):
        if self.proj_type == ProjectionType.normal or self.proj_type == 'normal':
            self.proj_matrix.normal_(generator=self.generator)
        elif self.proj_type == ProjectionType.rademacher or self.proj_type == 'rademacher':
            self.proj_matrix.bernoulli_(p=0.5, generator=self.generator)
            # going from Bernoulli {0, 1} to Rademacher {-1, 1}
            self.proj_matrix *= 2.
            self.proj_matrix -= 1.
        else:
            raise KeyError(f'Projection type {self.proj_type} not recognized.')

    def project(self, grads: Tensor, model_id: int) -> Tensor:
        if model_id != self.model_id:
            self.model_id = model_id
            self.generator = self.generator.manual_seed(self.seed + 10e4 * self.model_id)
            self.generate_sketch_matrix()  # updates self.proj_matrix

        return grads @ self.proj_matrix


class BasicProjector(AbstractProjector):
    """
    A simple block-wise implementation of the projection. The projection matrix
    is generated on-device in blocks. The accumulated result across blocks is
    returned.

    Note: This class will be significantly slower and have a larger memory
    footprint than the CudaProjector. It is recommended that you use this method
    only if the CudaProjector is not available to you -- e.g. if you don't have
    a CUDA-enabled device with compute capability >=7.0 (see
    https://developer.nvidia.com/cuda-gpus).
    """
    def __init__(self, grad_dim: int, proj_dim: int, seed: int, proj_type:
                 ProjectionType, device, block_size: int = 200, dtype=ch.float32,
                 model_id=0) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

        self.block_size = min(self.proj_dim, block_size)
        self.num_blocks = math.ceil(self.proj_dim / self.block_size)
        self.dtype = dtype
        self.proj_type = proj_type
        self.model_id = model_id

        self.proj_matrix = ch.empty(self.grad_dim, self.block_size,
                                    dtype=self.dtype,
                                    device=self.device)

        self.generator = ch.Generator(device=self.device)

        self.get_generator_states()
        self.generate_sketch_matrix(self.generator_states[0])

    def get_generator_states(self):
        self.generator_states = []
        self.seeds = []
        self.jl_size = self.proj_matrix.numel()

        for i in range(self.num_blocks):
            s = self.seed + int(1e3) * i + int(1e5) * self.model_id
            self.seeds.append(s)
            self.generator = self.generator.manual_seed(s)
            self.generator_states.append(self.generator.get_state())

    def generate_sketch_matrix(self, generator_state):
        self.generator.set_state(generator_state)
        if self.proj_type == ProjectionType.normal or self.proj_type == 'normal':
            self.proj_matrix.normal_(generator=self.generator)
        elif self.proj_type == ProjectionType.rademacher or self.proj_type == 'rademacher':
            self.proj_matrix.bernoulli_(p=0.5, generator=self.generator)
            self.proj_matrix *= 2.
            self.proj_matrix -= 1.
        else:
            raise KeyError(f'Projection type {self.proj_type} not recognized.')

    def project(self, grads: Tensor, model_id: int) -> Tensor:
        sketch = ch.zeros(size=(grads.size(0), self.proj_dim),
                          dtype=self.dtype, device=self.device)

        if model_id != self.model_id:
            self.model_id = model_id
            self.get_generator_states()  # regenerate random seeds for new model_id
            if self.num_blocks == 1:
                self.generate_sketch_matrix(self.generator_states[0])

        if self.num_blocks == 1:
            ch.matmul(grads.data, self.proj_matrix, out=sketch)
        else:
            for ind in range(self.num_blocks):
                self.generate_sketch_matrix(self.generator_states[ind])

                st = ind * self.block_size
                ed = min((ind + 1) * self.block_size, self.proj_dim)
                sketch[:, st:ed] = grads.type(self.dtype) @ self.proj_matrix[:, :(ed - st)]
        return sketch.type(grads.dtype)


class CudaProjector(AbstractProjector):
    """
    A performant implementation of the projection for CUDA with compute
    capability >= 7.0.
    """
    def __init__(self, grad_dim: int, proj_dim: int, seed: int, proj_type:
                 ProjectionType, device, *args, **kwargs) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

        if isinstance(device, str):
            device = ch.device(device)

        if device.type != 'cuda':
            err = "CudaProjector only works on a CUDA device; Either switch to a CUDA device, or use the BasicProjector"
            raise ValueError(err)

        self.num_sms = ch.cuda.get_device_properties(device.index).multi_processor_count

        try:
            import fast_jl
            # test run to catch at init time if projection goes through
            fast_jl.project_rademacher_8(ch.zeros(8, 1_000, device='cuda'), 512, 0, self.num_sms)
        except ImportError:
            err = "You should make sure to install the CUDA projector for traker (called fast_jl).\
                  See the installation FAQs for more details."
            raise ModuleNotFoundError(err)

    def project(self, grads: Tensor, model_id: int) -> Tensor:
        batch_size = grads.shape[0]
        effective_batch_size = 32
        if batch_size <= 8:
            effective_batch_size = 8
        elif batch_size <= 16:
            effective_batch_size = 16

        function_name = f"project_{self.proj_type.value}_{effective_batch_size}"
        import fast_jl
        fn = getattr(fast_jl, function_name)
        return fn(grads, self.proj_dim, self.seed + int(1e4) * model_id, self.num_sms)
