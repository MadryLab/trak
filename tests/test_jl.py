import pytest
import math
from itertools import product
import numpy as np
import torch as ch
from torch import testing

from trak.projectors import CudaProjector, ProjectionType, ChunkedCudaProjector
BasicProjector = CudaProjector

MAX_BATCH_SIZE = 32
# TEST CASES 1
PARAM = list(product([123],  # seed
                     [ProjectionType.rademacher],  # proj type
                     [ch.float32],  # dtype
                     [
                        # tests that shows relation with MAXINT32
                        #  (8, 180645096), # pass: np.prod(shape) < np.iinfo(np.int32).max
                        #  (16, 180645096), # pass: np.prod(shape) > np.iinfo(np.int32).max
                        #  (31, 180645096), # fail: np.prod(shape) > np.iinfo(np.int32).max
                        #  (32, 180645096), # fail: np.prod(shape) > np.iinfo(np.int32).max
                        #  (33, 180645096), # pass: np.prod(shape) > np.iinfo(np.int32).max
                        #  (48, 180645096), # pass: np.prod(shape) > np.iinfo(np.int32).max
                        #  (50, 180645096), # pass: np.prod(shape) > np.iinfo(np.int32).max
                         (2, 780645096), # fail: np.prod(shape) > np.iinfo(np.int32).max
                        #  (8, 780645096), # fail: np.prod(shape) > np.iinfo(np.int32).max
                     ],  # input shape
                     [15_360],  # proj dim
                     ))

# TEST CASES 2
# PARAM = list(product([123],  # seed
#                      [ProjectionType.rademacher],  # proj type
#                      [ch.float32],  # dtype
#                      [
#                         # tests that shows relation with MAXINT32
#                         #  (1, 780645096), # pass: np.prod(shape) < np.iinfo(np.int32).max
#                         #  (5, 780645096), # pass: np.prod(shape) > np.iinfo(np.int32).max
#                         #  (6, 780645096), # pass: np.prod(shape) > np.iinfo(np.int32).max
#                         #  (7, 780645096), # fail: np.prod(shape) > np.iinfo(np.int32).max
#                         #  (8, 780645096), # fail: np.prod(shape) > np.iinfo(np.int32).max
#                      ],  # input shape
#                      [4_096],  # proj dim
#                     #  [15_360],  # proj dim same results here
#                      ))

@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_seed_consistency(seed,
                          proj_type,
                          dtype,
                          proj_dim,
                          input_shape,
                          ):
    """
    Check that re-running the same projection with the same seed
    leads to the same result.
    """

    g = testing.make_tensor(*input_shape, device='cuda:0', dtype=dtype)
    proj = BasicProjector(grad_dim=input_shape[-1],
                          proj_dim=proj_dim,
                          proj_type=proj_type,
                          seed=seed,
                          device='cuda:0',
                          dtype=dtype,
                          max_batch_size=MAX_BATCH_SIZE
                          )

    result = proj.project(g, model_id=0)
    result_again = proj.project(g, model_id=0)
    testing.assert_close(result, result_again, equal_nan=True)


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_seed_consistency_2(seed,
                            proj_type,
                            dtype,
                            proj_dim,
                            input_shape,
                            ):
    """
    Check that re-initializing the class and re-running the same projection
    with the same seed leads to the same result.
    """

    g = testing.make_tensor(*input_shape, device='cuda:0', dtype=dtype)
    proj = BasicProjector(grad_dim=input_shape[-1],
                          proj_dim=proj_dim,
                          proj_type=proj_type,
                          seed=seed,
                          device='cuda:0',
                          dtype=dtype,
                          max_batch_size=MAX_BATCH_SIZE
                          )

    result = proj.project(g, model_id=0)

    proj_again = BasicProjector(grad_dim=input_shape[-1],
                                proj_dim=proj_dim,
                                proj_type=proj_type,
                                seed=seed,
                                device='cuda:0',
                                dtype=dtype,
                                max_batch_size=MAX_BATCH_SIZE
                                )
    result_again = proj_again.project(g, model_id=0)
    testing.assert_close(result, result_again, equal_nan=True)


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_norm_preservation(seed,
                           proj_type,
                           dtype,
                           proj_dim,
                           input_shape,
                           ):
    """
    Check that norms of differences are approximately preserved.
    """
    g = testing.make_tensor(*input_shape, device='cuda:0', dtype=dtype)
    proj = BasicProjector(grad_dim=input_shape[-1],
                          proj_dim=proj_dim,
                          proj_type=proj_type,
                          seed=seed,
                          device='cuda:0',
                          dtype=dtype,
                          max_batch_size=MAX_BATCH_SIZE
                          )

    p = proj.project(g, model_id=0)

    delta = 0.05
    eps = np.sqrt(np.log(1 / delta) / proj_dim)
    num_trials = 100
    num_successes = 0

    for _ in range(num_trials):
        i, j = np.random.choice(range(g.shape[0]), size=2)
        n = (g[i] - g[j]).norm()
        # assuming for the test that the norm of each column
        # of the projection matrix has norm sqrt(n)
        # (true for rademacher and approx true for gaussian)
        pn = (p[i] - p[j]).norm() / np.sqrt(input_shape[-1])
        res = (n - pn).cpu().abs().item()
        # 35 is an arbitrary constant
        # if NaN, just give up and count as success
        if math.isinf(res):
            print('aaaaaa')
        num_successes += int(res <= 35 * eps * n)
    assert num_successes >= num_trials * (1 - 3 * delta)  # leeway with 2 *


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_prod_preservation(seed,
                           proj_type,
                           dtype,
                           proj_dim,
                           input_shape,
                           ):
    """
    Check that dot products are approximately preserved.
    """
    g = testing.make_tensor(*input_shape, device='cuda:0', dtype=dtype)
    proj = BasicProjector(grad_dim=input_shape[-1],
                          proj_dim=proj_dim,
                          proj_type=proj_type,
                          seed=seed,
                          device='cuda:0',
                          dtype=dtype,
                          max_batch_size=MAX_BATCH_SIZE
                          )

    # check that things break with a garbage matrix
    # (making sure the constant 15 is reasonable)
    # proj.proj_matrix = ch.empty_like(proj.proj_matrix)

    p = proj.project(g, model_id=0)

    delta = 0.2
    eps = np.sqrt(np.log(1 / delta) / proj_dim)
    num_trials = 100
    num_successes = 0

    for _ in range(num_trials):
        i, j = np.random.choice(range(g.shape[0]), size=2)
        n = (g[i] @ g[j])
        pn = ((p[i] / np.sqrt(input_shape[-1])) @ (p[j] / input_shape[-1]))
        res = (n.abs() - pn.abs()).cpu().abs().item()
        t = (50 * np.sqrt(proj_dim) * eps * n).abs().item()
        # if NaN, just give up and count as success
        num_successes += max(int(res <= t), math.isinf(res), math.isinf(t))

    assert num_successes >= num_trials * (1 - 2 * delta)


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_single_nonzero_feature(seed,
                                proj_type,
                                dtype,
                                proj_dim,
                                input_shape,
                                ):
    """
    Check that output takes into account every feature.
    """
    g = ch.zeros(*input_shape, device='cuda:0', dtype=dtype)
    for ind in range(input_shape[0]):
        coord = np.random.choice(range(input_shape[1]))
        val = ch.randn(1)
        g[ind, coord] = val.item()

    proj = BasicProjector(grad_dim=input_shape[-1],
                          proj_dim=proj_dim,
                          proj_type=proj_type,
                          seed=seed,
                          device='cuda:0',
                          dtype=dtype,
                          max_batch_size=MAX_BATCH_SIZE
                          )
    p = proj.project(g, model_id=0)
    assert (~ch.isclose(p, ch.zeros_like(p))).all().item()


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_first_nonzero_feature(seed,
                               proj_type,
                               dtype,
                               proj_dim,
                               input_shape,
                               ):
    """
    Check that output takes into account first features.
    """
    g = ch.zeros(*input_shape, device='cuda:0', dtype=dtype)
    g[:, 0] = 1.

    proj = BasicProjector(grad_dim=input_shape[-1],
                          proj_dim=proj_dim,
                          proj_type=proj_type,
                          seed=seed,
                          device='cuda:0',
                          dtype=dtype,
                          max_batch_size=MAX_BATCH_SIZE
                          )
    p = proj.project(g, model_id=0)
    assert (~ch.isclose(p, ch.zeros_like(p))).all().item()


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_last_nonzero_feature(seed,
                              proj_type,
                              dtype,
                              proj_dim,
                              input_shape,
                              ):
    """
    Check that output takes into account last features.
    """
    g = ch.zeros(*input_shape, device='cuda:0', dtype=dtype)
    g[:, -1] = 1.

    proj = BasicProjector(grad_dim=input_shape[-1],
                          proj_dim=proj_dim,
                          proj_type=proj_type,
                          seed=seed,
                          device='cuda:0',
                          dtype=dtype,
                          max_batch_size=MAX_BATCH_SIZE
                          )
    p = proj.project(g, model_id=0)
    assert (~ch.isclose(p, ch.zeros_like(p))).all().item()


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
    midpoint = g.size(0) // 2
    for i in range(midpoint):
        g[i] = g[-(i+1)]

    proj = BasicProjector(grad_dim=input_shape[-1],
                          proj_dim=proj_dim,
                          proj_type=proj_type,
                          seed=seed,
                          device='cuda:0',
                          dtype=dtype,
                          max_batch_size=MAX_BATCH_SIZE)

    p = proj.project(g, model_id=0)

    assert all([ch.allclose(p[i], p[-(i+1)]) for i in range(midpoint)])

@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_chunked_projection(seed,
                            proj_type,
                            dtype,
                            proj_dim,
                            input_shape,
                            ):
    """
    Check that output is the same for the same features (when large input)
    """
    g = testing.make_tensor(*input_shape, device='cuda:0', dtype=dtype)

    assert np.prod(g.shape) > np.iinfo(np.uint32).max, 'Input not too large, check `test_same_features` instead'

    bs, num_params = input_shape
    max_chunk_size = np.iinfo(np.uint32).max // bs
    num_chunks = np.ceil(num_params / max_chunk_size).astype('int32')
    g_chunks = ch.chunk(g, num_chunks, dim=1)

    # naive chunking
    naive_projectors = [
        BasicProjector(grad_dim=x.size(-1),
                        proj_dim=proj_dim,
                        proj_type=proj_type,
                        seed=seed + i,
                        device='cuda:0',
                        dtype=dtype,
                        max_batch_size=MAX_BATCH_SIZE
                        )

        for i, x in enumerate(g_chunks)
    ]


    all_projs = [proj_i.project(g_i.contiguous(), model_id=0) for i, (g_i, proj_i) in enumerate(zip(g_chunks, naive_projectors))]
    naive_projection = sum(all_projs)

    # fast projection
    chunk_projectors = [
        BasicProjector(grad_dim=x.size(-1),
                        proj_dim=proj_dim,
                        proj_type=proj_type,
                        seed=seed + i,
                        device='cuda:0',
                        dtype=dtype,
                        max_batch_size=MAX_BATCH_SIZE
                        )

        for i, x in enumerate(g_chunks)
    ]

    params_per_chunk = [x.size(1) for x in g_chunks]
    chunked_projector = ChunkedCudaProjector(chunk_projectors,
                                            max_chunk_size,
                                            params_per_chunk,
                                            bs,
                                            'cuda:0',
                                            dtype)

    g_values_dict = {i: x for i, x in enumerate(g_chunks)}
    chunked_projection = chunked_projector.project(g_values_dict, model_id=0)

    assert ch.allclose(naive_projection, chunked_projection)

@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_same_features_chunked(seed,
                               proj_type,
                               dtype,
                               proj_dim,
                               input_shape,
                               ):
    """
    Check that output is the same for the same features
    """

    g = testing.make_tensor(*input_shape, device='cuda:0', dtype=dtype)
    midpoint = g.size(0) // 2
    for i in range(midpoint):
        g[i] = g[-(i+1)]

    bs, num_params = input_shape
    max_chunk_size = np.iinfo(np.uint32).max // bs
    num_chunks = np.ceil(num_params / max_chunk_size).astype('int32')
    g_chunks = ch.chunk(g, num_chunks, dim=1)

    chunk_projectors = [
        BasicProjector(grad_dim=x.size(-1),
                        proj_dim=proj_dim,
                        proj_type=proj_type,
                        seed=seed + i,
                        device='cuda:0',
                        dtype=dtype,
                        max_batch_size=MAX_BATCH_SIZE
                        )

        for i, x in enumerate(g_chunks)
    ]

    params_per_chunk = [x.size(1) for x in g_chunks]
    chunked_projector = ChunkedCudaProjector(chunk_projectors,
                                            max_chunk_size,
                                            params_per_chunk,
                                            bs,
                                            'cuda:0',
                                            dtype)

    g_values_dict = {i: x for i, x in enumerate(g_chunks)}
    p = chunked_projector.project(g_values_dict, model_id=0)

    assert all([ch.allclose(p[i], p[-(i+1)]) for i in range(midpoint)])

@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_orthogonality(seed,
                       proj_type,
                       dtype,
                       proj_dim,
                       input_shape,
                       ):
    """
    Check that orthgonality of inputs is approximately presereved whp
    """
    if input_shape[0] == 1:
        pass
    else:
        proj = BasicProjector(grad_dim=input_shape[-1],
                              proj_dim=proj_dim,
                              proj_type=proj_type,
                              seed=seed,
                              device='cuda:0',
                              dtype=dtype,
                              max_batch_size=MAX_BATCH_SIZE
                              )

        num_successes = 0
        num_trials = 100
        for _ in range(num_trials):
            g = testing.make_tensor(*input_shape, device='cuda:0', dtype=dtype)
            g[-1] -= g[0] @ g[-1] / (g[0].norm() ** 2) * g[0]
            p = proj.project(g, model_id=0)
            if p[0] @ p[-1] < 1e-3:
                num_successes += 1
        assert num_successes > 0.33 * num_trials

