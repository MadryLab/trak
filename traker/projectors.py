from abc import ABC, abstractmethod
from enum import Enum
import math
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
    def project(self, grads: Tensor, model_id: int) -> Tensor:
        ...


class BasicSingleBlockProjector(AbstractProjector):
    """
    A bare-bones implementation of the projection, which is (extremely)
    inefficient in terms of both time and memory footrpint.
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
    An implementation of the projection which performs the
    matmul blockwise if needed.
    """
    def __init__(self, grad_dim: int, proj_dim: int, seed: int, proj_type:
                 ProjectionType, device, num_blocks: int=2, dtype=ch.float16,
                 model_id=0) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

        self.num_blocks = num_blocks
        self.block_size = math.ceil(self.proj_dim / self.num_blocks)
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
                sketch[:, st:ed] = grads @ self.proj_matrix[:, :(ed - st)]
        return sketch


class CudaProjector(AbstractProjector):
    """
    An implementation of the project for cuda (with compute capability >= 7.0)
    """
    def __init__(self, grad_dim: int, proj_dim: int, seed: int, proj_type:
                 ProjectionType, device, *args, **kwargs) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

        if isinstance(device, str):
            device = ch.device(device)

        if device.type != 'cuda':
            raise ValueError("CudaProjector only works on a cuda device; Either switch to a cuda device, or use the BasicProjector")

        self.num_sms = ch.cuda.get_device_properties(device.index).multi_processor_count

        try:
            import fast_jl
        except ImportError:
            raise ModuleNotFoundError("You should make sure you install the cuda projetor for traker (called fast_jl)")

    def project(self, grads: Tensor, model_id: int) -> Tensor:
        batch_size = grads.shape[0]
        ebs = 32
        if batch_size <= 8:
            ebs = 8
        elif batch_size <= 16:
            ebs = 16

        function_name = f"project_{self.proj_type.value}_{ebs}"
        import fast_jl
        fn = getattr(fast_jl, function_name)
        return fn(grads, self.proj_dim, self.seed + int(1e4) * model_id, self.num_sms)
