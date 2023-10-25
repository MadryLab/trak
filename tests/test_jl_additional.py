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
                         (8, 180645096),  # pass: np.prod(shape) < np.iinfo(np.int32).max
                         (16, 180645096),  # pass: np.prod(shape) > np.iinfo(np.int32).max
                         (31, 180645096),  # fail: np.prod(shape) > np.iinfo(np.int32).max
                         (32, 180645096),  # fail: np.prod(shape) > np.iinfo(np.int32).max
                         (33, 180645096),  # pass: np.prod(shape) > np.iinfo(np.int32).max
                         (48, 180645096),  # pass: np.prod(shape) > np.iinfo(np.int32).max
                         (50, 180645096),  # pass: np.prod(shape) > np.iinfo(np.int32).max
                     ],  # input shape
                     [15_360],  # proj dim
                     ))

# TEST CASES 2
PARAM = list(product([123],  # seed
                     [ProjectionType.rademacher],  # proj type
                     [ch.float32],  # dtype
                     [
                         (1, 780645096),  # pass: np.prod(shape) < np.iinfo(np.int32).max
                         (5, 780645096),  # pass: np.prod(shape) > np.iinfo(np.int32).max
                         (6, 780645096),  # pass: np.prod(shape) > np.iinfo(np.int32).max
                         (7, 780645096),  # fail: np.prod(shape) > np.iinfo(np.int32).max
                         (8, 780645096),  # fail: np.prod(shape) > np.iinfo(np.int32).max
                     ],  # input shape
                     [4_096],  # proj dim
                     ))


# TEST CASES 3 (ONLY for test_same_features_diff_sms)
PARAM = list(product([123],  # seed
                     [ProjectionType.rademacher],  # proj type
                     [ch.float32],  # dtype
                     [
                         (32, 100_000),
                     ],  # input shape
                     [4_096],  # proj dim
                     ))

@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_same_features(seed,
                       proj_type,
                       dtype,
                       proj_dim,
                       input_shape,
                       ):
    """
    Check that output is the same for the same features
    """
    g = testing.make_tensor(*input_shape, device='cuda:0', dtype=dtype)
    g[-1] = g[0]

    proj = CudaProjector(grad_dim=input_shape[-1],
                         proj_dim=proj_dim,
                         proj_type=proj_type,
                         seed=seed,
                         device='cuda:0',
                         dtype=dtype,
                         max_batch_size=MAX_BATCH_SIZE
                         )
    p = proj.project(g, model_id=0)

    assert ch.allclose(p[0], p[-1])

@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_same_features_diff_sms(seed,
                                proj_type,
                                dtype,
                                proj_dim,
                                input_shape,
                                ):
    """
    Check that output is the same for the same features
    """
    g = testing.make_tensor(*input_shape, device='cuda:0', dtype=dtype)


    # project with all SMs available
    proj_full_sms = CudaProjector(grad_dim=input_shape[-1],
                                  proj_dim=proj_dim,
                                  proj_type=proj_type,
                                  seed=seed,
                                  device='cuda:0',
                                  dtype=dtype,
                                  max_batch_size=MAX_BATCH_SIZE
                                  )
    p_full_sms = proj_full_sms.project(g, model_id=0)

    # project with half SMs available
    proj_half_sms = CudaProjector(grad_dim=input_shape[-1],
                                  proj_dim=proj_dim,
                                  proj_type=proj_type,
                                  seed=seed,
                                  device='cuda:0',
                                  dtype=dtype,
                                  max_batch_size=MAX_BATCH_SIZE
                                  )

    proj_half_sms.num_sms = max(proj_half_sms.num_sms // 2, 1)
    p_half_sms = proj_half_sms.project(g, model_id=0)

    assert ch.allclose(p_full_sms, p_half_sms)
