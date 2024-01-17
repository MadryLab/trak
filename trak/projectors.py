"""
Projectors are used to project gradients to a lower-dimensional space. This 1) allows
us to compute TRAK scores in a *much* more efficient manner, and 2) turns out to be
act as a useful regularizer (see Appendix E.1 in our paper).

Here, we provide four implementations of the projector:
- :class:`NoOpProjector` (no-op)
- :class:`BasicSingleBlockProjector` (bare-bones, inefficient implementation)
- :class:`BasicProjector` (block-wise implementation)
- :class:`CudaProjector` (a fast implementation with a custom CUDA kernel)
"""
from abc import ABC, abstractmethod
from typing import Union
from enum import Enum
import math
from torch import Tensor
import torch

from .utils import vectorize


ch = torch


class ProjectionType(str, Enum):
    normal: str = "normal"
    rademacher: str = "rademacher"


class AbstractProjector(ABC):
    """Implementations of the Projector class must implement the
    :meth:`AbstractProjector.project` method, which takes in model gradients and
    returns
    """

    @abstractmethod
    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: Union[str, ProjectionType],
        device: Union[str, torch.device],
    ) -> None:
        """Initializes hyperparameters for the projection.

        Args:
            grad_dim (int):
                number of parameters in the model (dimension of the gradient
                vectors)
            proj_dim (int):
                dimension after the projection
            seed (int):
                random seed for the generation of the sketching (projection)
                matrix
            proj_type (Union[str, ProjectionType]):
                the random projection (JL transform) guearantees that distances
                will be approximately preserved for a variety of choices of the
                random matrix (see e.g. https://arxiv.org/abs/1411.2404). Here,
                we provide an implementation for matrices with iid Gaussian
                entries and iid Rademacher entries.
            device (Union[str, torch.device]):
                CUDA device to use

        """
        self.grad_dim = grad_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.proj_type = proj_type
        self.device = device

    @abstractmethod
    def project(self, grads: Tensor, model_id: int) -> Tensor:
        """Performs the random projection. Model ID is included
        so that we generate different projection matrices for every
        model ID.

        Args:
            grads (Tensor): a batch of gradients to be projected
            model_id (int): a unique ID for a checkpoint

        Returns:
            Tensor: the projected gradients
        """

    def free_memory(self):
        """Frees up memory used by the projector."""


class NoOpProjector(AbstractProjector):
    """
    A projector that returns the gradients as they are, i.e., implements
    :code:`projector.project(grad) = grad`.
    """

    def __init__(
        self,
        grad_dim: int = 0,
        proj_dim: int = 0,
        seed: int = 0,
        proj_type: Union[str, ProjectionType] = "na",
        device: Union[str, torch.device] = "cuda",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

    def project(self, grads: Tensor, model_id: int) -> Tensor:
        """A no-op method.

        Args:
            grads (Tensor): a batch of gradients to be projected
            model_id (int): a unique ID for a checkpoint

        Returns:
            Tensor: the (non-)projected gradients
        """
        if isinstance(grads, dict):
            grads = vectorize(grads, device=self.device)
        return grads

    def free_memory(self):
        """A no-op method."""
        pass


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

    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: ProjectionType,
        device,
        dtype=ch.float32,
        model_id=0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

        self.model_id = model_id
        self.proj_type = proj_type
        self.generator = ch.Generator(device=self.device)
        self.generator = self.generator.manual_seed(
            self.seed + int(1e4) * self.model_id
        )
        self.dtype = dtype

        self.proj_matrix = ch.empty(
            self.grad_dim, self.proj_dim, dtype=self.dtype, device=self.device
        )

        self.proj_matrix_available = True

        self.generate_sketch_matrix()  # updates self.proj_matrix

    def free_memory(self):
        del self.proj_matrix
        self.proj_matrix_available = False

    def generate_sketch_matrix(self):
        if not self.proj_matrix_available:
            self.proj_matrix = ch.empty(
                self.grad_dim, self.proj_dim, dtype=self.dtype, device=self.device
            )
            self.proj_matrix_available = True

        if self.proj_type == ProjectionType.normal or self.proj_type == "normal":
            self.proj_matrix.normal_(generator=self.generator)
        elif (
            self.proj_type == ProjectionType.rademacher
            or self.proj_type == "rademacher"
        ):
            self.proj_matrix.bernoulli_(p=0.5, generator=self.generator)
            # going from Bernoulli {0, 1} to Rademacher {-1, 1}
            self.proj_matrix *= 2.0
            self.proj_matrix -= 1.0
        else:
            raise KeyError(f"Projection type {self.proj_type} not recognized.")

    def project(self, grads: Tensor, model_id: int) -> Tensor:
        if isinstance(grads, dict):
            grads = vectorize(grads, device=self.device)

        grads = grads.to(dtype=self.dtype)
        if model_id != self.model_id:
            self.model_id = model_id
            self.generator = self.generator.manual_seed(
                self.seed + int(1e4) * self.model_id
            )
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

    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: ProjectionType,
        device: torch.device,
        block_size: int = 100,
        dtype: torch.dtype = ch.float32,
        model_id=0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

        self.block_size = min(self.proj_dim, block_size)
        self.num_blocks = math.ceil(self.proj_dim / self.block_size)
        self.dtype = dtype
        self.proj_type = proj_type
        self.model_id = model_id

        self.proj_matrix = ch.empty(
            self.grad_dim, self.block_size, dtype=self.dtype, device=self.device
        )

        self.proj_matrix_available = True

        self.generator = ch.Generator(device=self.device)

        self.get_generator_states()
        self.generate_sketch_matrix(self.generator_states[0])

    def free_memory(self):
        del self.proj_matrix
        self.proj_matrix_available = False

    def get_generator_states(self):
        self.generator_states = []
        self.seeds = []
        self.jl_size = self.grad_dim * self.block_size

        for i in range(self.num_blocks):
            s = self.seed + int(1e3) * i + int(1e5) * self.model_id
            self.seeds.append(s)
            self.generator = self.generator.manual_seed(s)
            self.generator_states.append(self.generator.get_state())

    def generate_sketch_matrix(self, generator_state):
        if not self.proj_matrix_available:
            self.proj_matrix = ch.empty(
                self.grad_dim, self.block_size, dtype=self.dtype, device=self.device
            )
            self.proj_matrix_available = True

        self.generator.set_state(generator_state)
        if self.proj_type == ProjectionType.normal or self.proj_type == "normal":
            self.proj_matrix.normal_(generator=self.generator)
        elif (
            self.proj_type == ProjectionType.rademacher
            or self.proj_type == "rademacher"
        ):
            self.proj_matrix.bernoulli_(p=0.5, generator=self.generator)
            self.proj_matrix *= 2.0
            self.proj_matrix -= 1.0
        else:
            raise KeyError(f"Projection type {self.proj_type} not recognized.")

    def project(self, grads: Tensor, model_id: int) -> Tensor:
        if isinstance(grads, dict):
            grads = vectorize(grads, device=self.device)
        grads = grads.to(dtype=self.dtype)
        sketch = ch.zeros(
            size=(grads.size(0), self.proj_dim), dtype=self.dtype, device=self.device
        )

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
                sketch[:, st:ed] = (
                    grads.type(self.dtype) @ self.proj_matrix[:, : (ed - st)]
                )
        return sketch.type(grads.dtype)


class CudaProjector(AbstractProjector):
    """
    A performant implementation of the projection for CUDA with compute
    capability >= 7.0.
    """

    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: ProjectionType,
        device,
        max_batch_size: int,
        *args,
        **kwargs,
    ) -> None:
        """

        Args:
            grad_dim (int):
                Number of parameters
            proj_dim (int):
                Dimension we project *to* during the projection step
            seed (int):
                Random seed
            proj_type (ProjectionType):
                Type of randomness to use for projection matrix (rademacher or normal)
            device:
                CUDA device
            max_batch_size (int):
                Explicitly constraints the batch size the CudaProjector is going
                to use for projection. Set this if you get a 'The batch size of
                the CudaProjector is too large for your GPU' error. Must be
                either 8, 16, or 32.

        Raises:
            ValueError:
                When attempting to use this on a non-CUDA device
            ModuleNotFoundError:
                When fast_jl is not installed

        """
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)
        self.max_batch_size = max_batch_size

        if isinstance(device, str):
            device = ch.device(device)

        if device.type != "cuda":
            err = "CudaProjector only works on a CUDA device; Either switch to a CUDA device, or use the BasicProjector"
            raise ValueError(err)

        self.num_sms = ch.cuda.get_device_properties(device.index).multi_processor_count

        try:
            import fast_jl

            # test run to catch at init time if projection goes through
            fast_jl.project_rademacher_8(
                ch.zeros(8, 1_000, device="cuda"), 512, 0, self.num_sms
            )
        except ImportError:
            err = "You should make sure to install the CUDA projector for traker (called fast_jl).\
                  See the installation FAQs for more details."
            raise ModuleNotFoundError(err)

    def project(
        self,
        grads: Union[dict, Tensor],
        model_id: int,
    ) -> Tensor:
        if isinstance(grads, dict):
            grads = vectorize(grads, device=self.device)

        batch_size = grads.shape[0]

        effective_batch_size = 32
        if batch_size <= 8:
            effective_batch_size = 8
        elif batch_size <= 16:
            effective_batch_size = 16

        effective_batch_size = min(self.max_batch_size, effective_batch_size)

        function_name = f"project_{self.proj_type.value}_{effective_batch_size}"
        import fast_jl

        fn = getattr(fast_jl, function_name)

        try:
            result = fn(
                grads, self.proj_dim, self.seed + int(1e4) * model_id, self.num_sms
            )
        except RuntimeError as e:
            if "CUDA error: too many resources requested for launch" in str(e):
                # provide a more helpful error message
                raise RuntimeError(
                    (
                        "The batch size of the CudaProjector is too large for your GPU. "
                        "Reduce it by using the proj_max_batch_size argument of the TRAKer.\nOriginal error:"
                    )
                )
            else:
                raise e

        return result

    def free_memory(self):
        """A no-op method."""
        pass


class ChunkedCudaProjector:
    def __init__(
        self,
        projector_per_chunk: list,
        max_chunk_size: int,
        params_per_chunk: list,
        feat_bs: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.projector_per_chunk = projector_per_chunk
        self.proj_dim = self.projector_per_chunk[0].proj_dim
        self.proj_type = self.projector_per_chunk[0].proj_type
        self.params_per_chunk = params_per_chunk

        self.max_chunk_size = max_chunk_size
        self.feat_bs = feat_bs
        self.device = device
        self.dtype = dtype
        self.input_allocated = False

    def allocate_input(self):
        if self.input_allocated:
            return

        self.ch_input = ch.zeros(
            size=(self.feat_bs, self.max_chunk_size),
            device=self.device,
            dtype=self.dtype,
        )

        self.input_allocated = True

    def free_memory(self):
        if not self.input_allocated:
            return

        del self.ch_input
        self.input_allocated = False

    def project(self, grads, model_id):
        self.allocate_input()
        ch_output = ch.zeros(
            size=(self.feat_bs, self.proj_dim), device=self.device, dtype=self.dtype
        )
        pointer = 0
        # iterate over params, keep a counter of params so far, and when prev
        # chunk reaches max_chunk_size, project and accumulate
        projector_index = 0
        for i, p in enumerate(grads.values()):
            if len(p.shape) < 2:
                p_flat = p.data.unsqueeze(-1)
            else:
                p_flat = p.data.flatten(start_dim=1)

            param_size = p_flat.size(1)
            if pointer + param_size > self.max_chunk_size:
                # fill remaining entries with 0
                assert pointer == self.params_per_chunk[projector_index]
                # project and accumulate
                ch_output.add_(
                    self.projector_per_chunk[projector_index].project(
                        self.ch_input[:, :pointer].contiguous(),
                        model_id=model_id,
                    )
                )
                # reset counter
                pointer = 0
                projector_index += 1

            # continue accumulation
            actual_bs = min(self.ch_input.size(0), p_flat.size(0))
            self.ch_input[:actual_bs, pointer : pointer + param_size].copy_(p_flat)
            pointer += param_size

        # at the end, we need to project remaining items
        # fill remaining entries with 0
        assert pointer == self.params_per_chunk[projector_index]
        # project and accumulate
        ch_output[:actual_bs].add_(
            self.projector_per_chunk[projector_index].project(
                self.ch_input[:actual_bs, :pointer].contiguous(),
                model_id=model_id,
            )
        )

        return ch_output[:actual_bs]
