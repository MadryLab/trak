import os
import pytest
from itertools import product
import torch as ch
from torch import testing

from trak.projectors import CudaProjector, ProjectionType

MAX_BATCH_SIZE = 32

# TEST CASES 1
PARAM = list(product([123],  # seed
                     [ProjectionType.rademacher],  # proj type
                     [ch.float32],  # dtype
                     [
                         (32, 100_000),  # pass: np.prod(shape) < np.iinfo(np.int32).max
                     ],  # input shape
                     [4_096],  # proj dim
                     [108], # num sms
                     ))


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim, num_sms", PARAM)
@pytest.mark.cuda
def test_create_proj(seed,
                     proj_type,
                     dtype,
                     proj_dim,
                     input_shape,
                     num_sms,
                     ):
    """
    Compute the output for each GPU type
    """
    GPU_NAME = os.environ['GPU_NAME']
    print(f'GPU: {GPU_NAME}')

    if os.path.exists(f'./{GPU_NAME}.pt'):
        os.remove(f'./{GPU_NAME}.pt')

    g = testing.make_tensor(*input_shape, device='cuda:0', dtype=dtype)

    proj = CudaProjector(grad_dim=input_shape[-1],
                         proj_dim=proj_dim,
                         proj_type=proj_type,
                         seed=seed,
                         device='cuda:0',
                         dtype=dtype,
                         max_batch_size=MAX_BATCH_SIZE
                         )

    proj.num_sms = num_sms
    print(f'# Projector SMs: {proj.num_sms}')

    p = proj.project(g, model_id=0)

    ch.save(p.cpu(), f'./{GPU_NAME}.pt')


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim, num_sms", PARAM)
@pytest.mark.cuda
def test_same_proj(seed,
                   proj_type,
                   dtype,
                   proj_dim,
                   input_shape,
                   num_sms,
                   ):
    """
    Check that output is the same for different GPUs
    """

    proj_a100 = ch.load('./A100.pt')
    proj_h100 = ch.load('./H100.pt')

    assert ch.allclose(proj_a100, proj_h100), 'GPUs have different projection'