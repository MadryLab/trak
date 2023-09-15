from abc import ABC, abstractmethod
import numpy as np
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
    def __init__(self, dtype, device) -> None:
        self.dtype = dtype
        self.device = device

    @abstractmethod
    def get_xtx(self, grads: Tensor) -> Tensor:
        ...

    @abstractmethod
    def get_x_xtx_inv(self, grads: Tensor, xtx: Tensor) -> Tensor:
        ...

    @abstractmethod
    def get_scores(self, features: Tensor, target_grads: Tensor, accumulator: Tensor, target_dtype: type, **kwargs) -> Tensor:
        ...


class BasicSingleBlockScoreComputer(AbstractScoreComputer):
    """ A bare-bones implementation of :code:`ScoreComputer` that will likely
    OOM for almost all applications. Here for testing purposes only. Unless you
    have a good reason not to, you should use :func:`BasicScoreComputer`
    instead.
    """
    def __init__(self, dtype, device) -> None:
        super().__init__(dtype, device)

    def get_xtx(self, grads: Tensor) -> Tensor:
        return grads.T @ grads

    def get_x_xtx_inv(self, grads: Tensor, xtx: Tensor) -> Tensor:
        # torch.linalg.inv does not support float16
        return grads @ ch.linalg.inv(xtx.float()).to(self.dtype)

    def get_scores(self, features: Tensor, target_grads: Tensor, accumulator: Tensor, **kwargs) -> Tensor:
        return (features @ target_grads.T).cpu() + accumulator


class BasicScoreComputer(AbstractScoreComputer):
    """ An implementation of :code:`ScoreComputer` that computes matmuls in a
    block-wise manner.
    """
    def __init__(self, dtype, device, CUDA_MAX_DIM_SIZE: int = 100_000) -> None:
        """
        Args:
            device (Union[str, torch.device]): torch device to do matmuls on
            CUDA_MAX_DIM_SIZE (int, optional): Size of block for block-wise
            matmuls. Defaults to 100_000.
        """
        super().__init__(dtype, device)
        self.CUDA_MAX_DIM_SIZE = CUDA_MAX_DIM_SIZE

    def get_xtx(self, grads: Tensor) -> Tensor:
        self.proj_dim = grads.shape[1]
        result = ch.zeros(self.proj_dim, self.proj_dim, dtype=self.dtype, device=self.device)
        blocks = ch.split(grads, split_size_or_sections=self.CUDA_MAX_DIM_SIZE, dim=0)

        for block in blocks:
            result += block.T.to(self.device) @ block.to(self.device)

        return result

    def get_x_xtx_inv(self, grads: Tensor, xtx: Tensor) -> Tensor:
        blocks = ch.split(grads, split_size_or_sections=self.CUDA_MAX_DIM_SIZE, dim=0)
        xtx_inv = ch.linalg.inv(xtx.to(ch.float32))

        # center X^TX inverse a bit to avoid numerical issues when going to float16
        xtx_inv /= xtx_inv.abs().mean()

        xtx_inv = xtx_inv.to(self.dtype)

        result = ch.empty(grads.shape[0], xtx_inv.shape[1], dtype=self.dtype, device=self.device)
        for i, block in enumerate(blocks):
            start = i * self.CUDA_MAX_DIM_SIZE
            end = min(grads.shape[0], (i + 1) * self.CUDA_MAX_DIM_SIZE)
            result[start: end] = (block.to(self.device) @ xtx_inv)
        return result

    def get_scores(self, features: Tensor, target_grads: Tensor, accumulator: Tensor, target_dtype: type, **kwargs) -> Tensor:
        train_dim = features.shape[0]
        target_dim = target_grads.shape[0]

        if target_dim < self.CUDA_MAX_DIM_SIZE:
            return (features @ target_grads.T).to(device='cpu', dtype=target_dtype) + accumulator

        result = ch.empty(train_dim, target_dim, dtype=self.dtype, device=self.device)
        blocks = ch.split(target_grads, split_size_or_sections=self.CUDA_MAX_DIM_SIZE, dim=0)

        for i, block in enumerate(blocks):
            start = i * self.CUDA_MAX_DIM_SIZE
            end = min(target_grads.shape[0], (i + 1) * self.CUDA_MAX_DIM_SIZE)
            result[:, start: end] = features @ block.T

        return result.to(device='cpu', dtype=target_dtype) + accumulator

class BlockwiseScoreComputer(AbstractScoreComputer):
    """ An implementation of :code:`ScoreComputer` that computes matmuls in a
    block-wise manner.
    """
    def __init__(self, dtype, device, CUDA_MAX_DIM_SIZE: int = 500) -> None:
        """
        Args:
            device (Union[str, torch.device]): torch device to do matmuls on
            CUDA_MAX_DIM_SIZE (int, optional): Size of block for block-wise
            matmuls. Defaults to 100_000.
        """
        super().__init__(dtype, device)
        self.CUDA_MAX_DIM_SIZE = CUDA_MAX_DIM_SIZE

    def get_matrix_mult_blockwise(self, mat1: Tensor, mat2: Tensor, accumulator: Tensor, target_dtype: type, bs: int):

        s_mat1 = mat1.shape[0]
        s_mat2 = mat2.shape[0]

        bs = min(s_mat1, s_mat2, bs)

        # Copy the data in a pinned memory location to allow non-blocking
        # copies to the GPU
        mat1 = mat1.pin_memory()
        mat2 = mat2.pin_memory()

        # precompute all the blocks we will have to compute
        slices = []
        for i in range(int(np.ceil(s_mat1 / bs))):
            for j in range(int(np.ceil(s_mat2 / bs))):
                slices.append((slice(i * bs, (i + 1) * bs), slice(j * bs, (j + 1) * bs)))

        # Allocate memory for the final output.
        assert accumulator.shape == (s_mat1, s_mat2)

        # Output buffers pinned on the CPU to be able to collect data from the
        # GPU asynchronously
        # For each of our (2) cuda streams we need two output buffer, one
        # is currently written on with the next batch of result and the
        # second one is already finished and getting copied on the final output

        # If the size is not a multiple of batch size we need extra buffers
        # with the proper shapes
        outputs = [ch.zeros((bs, bs), dtype=target_dtype,
            device=mat1.device).pin_memory() for x in range(4)]
        left_bottom = s_mat1 % bs
        options = [outputs] # List of buffers we can potentially use
        if left_bottom:
            outputs_bottom = [ch.zeros((left_bottom, bs), dtype=target_dtype,
                device=mat1.device).pin_memory() for x in range(4)]
            options.append(outputs_bottom)
        left_right = s_mat2 % bs
        if left_right:
            outputs_right = [ch.zeros((bs, left_right), dtype=target_dtype,
                device=mat1.device).pin_memory() for x in range(4)]
            options.append(outputs_right)
        if left_right and left_bottom:
            outputs_corner = [ch.zeros((left_bottom, left_right), dtype=target_dtype,
                device=mat1.device).pin_memory() for x in range(4)]
            options.append(outputs_corner)

        streams = [ch.cuda.Stream() for x in range(2)]

        # The slice that was computed last and need to now copied onto the
        # final output
        previous_slice = None

        def find_buffer_for_shape(shape):
            for buff in options:
                if buff[0].shape == shape:
                    return buff
            return None

        for i, (slice_i, slice_j) in enumerate(slices):
            with ch.cuda.stream(streams[i % len(streams)]):
                # Copy the relevant blocks from CPU to the GPU asynchronously
                mat1_i = mat1[slice_i, :].cuda(non_blocking=True)
                mat2_j = mat2[slice_j, :].cuda(non_blocking=True)

                output_slice = mat1_i @ mat2_j.t() + accumulator[slice_i, slice_j].cuda(non_blocking=True)

                find_buffer_for_shape(output_slice.shape)[i % 4].copy_(output_slice, non_blocking=False)

            # Write the previous batch of data from the temporary buffer
            # onto the final one (note that this was done by the other stream
            # so we swap back to the other one
            with ch.cuda.stream(streams[(i + 1) % len(streams)]):
                if previous_slice is not None:
                    output_slice = accumulator[previous_slice[0], previous_slice[1]]
                    output_slice.copy_(find_buffer_for_shape(output_slice.shape)[(i - 1) % 4],
                            non_blocking=True)

            previous_slice = (slice_i, slice_j)

        # Wait for all the calculations/copies to be done
        ch.cuda.synchronize()

        # Copy the last chunk to the final result (from the appropriate buffer)
        output_slice = accumulator[previous_slice[0], previous_slice[1]]
        output_slice.copy_(find_buffer_for_shape(output_slice.shape)[i % 4],
                non_blocking=True)

        return accumulator

    def get_xtx(self, grads: Tensor) -> Tensor:
        self.proj_dim = grads.shape[1]

        result = ch.zeros(self.proj_dim, self.proj_dim, dtype=self.dtype, device=self.device)
        blocks = ch.split(grads, split_size_or_sections=self.CUDA_MAX_DIM_SIZE, dim=0)
        for block in blocks:
            result += block.T.to(self.device) @ block.to(self.device)

        return result

    def get_x_xtx_inv(self, grads: Tensor, xtx: Tensor, **kwargs) -> Tensor:
        blocks = ch.split(grads, split_size_or_sections=self.CUDA_MAX_DIM_SIZE, dim=0)
        xtx_inv = ch.linalg.inv(xtx.to(ch.float32))

        # center X^TX inverse a bit to avoid numerical issues when going to float16
        xtx_inv /= xtx_inv.abs().mean()

        xtx_inv = xtx_inv.to(self.dtype)

        result = ch.empty(grads.shape[0], xtx_inv.shape[1], dtype=self.dtype, device=self.device)
        for i, block in enumerate(blocks):
            start = i * self.CUDA_MAX_DIM_SIZE
            end = min(grads.shape[0], (i + 1) * self.CUDA_MAX_DIM_SIZE)
            result[start: end] = (block.to(self.device) @ xtx_inv)

        return result

    def get_scores(self, features: Tensor, target_grads: Tensor, accumulator: Tensor, target_dtype: type, **kwargs) -> Tensor:
        block_size = kwargs.get('block_size', 16_384)

        assert features.device == ch.device('cpu'), "Tensor `features` expected to be on cpu for blockwise multiplication"
        assert target_grads.device == ch.device('cpu'), "Tensor `target_grads` expected to be on cpu for blockwise multiplication"

        return self.get_matrix_mult_blockwise(features, target_grads, accumulator, target_dtype, bs=block_size)

SCORE_COMPUTERS = {
    'BasicSingleBlockScoreComputer': BasicSingleBlockScoreComputer,
    'BasicScoreComputer': BasicScoreComputer,
    'BlockwiseScoreComputer': BlockwiseScoreComputer,
}