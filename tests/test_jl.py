import pytest
import math
from itertools import product
import numpy as np
import torch as ch
from torch import testing

from trak.projectors import BasicProjector

PARAM = list(product([0, 1, 10**8], # seed
                     ['normal', 'rademacher'], # proj type
                     [ch.float16, ch.float32], # dtype
                     [
                      (1, 10_000),
                      (10, 10_000),
                      (100, 100_000),
                      ], # input shape
                     [1000], # proj dim
        ))

# will OOM for BasicProjector
PARAM_HEAVY = list(product([0, 1], # seed
                           ['normal', 'rademacher'], # proj type
                           [ch.float16, ch.float32], # dtype
                           [(1, int(1e10)),
                            (10, int(1e10)),
                            (100, int(1e10)),
                           ], # input shape
                           [20_000], # proj dim
        ))

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
                          dtype=dtype
                          )

    result = proj.project(g)
    result_again = proj.project(g)
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
                          dtype=dtype
                          )

    result = proj.project(g)

    proj_again = BasicProjector(grad_dim=input_shape[-1],
                                proj_dim=proj_dim,
                                proj_type=proj_type,
                                seed=seed,
                                device='cuda:0',
                                dtype=dtype)
    result_again = proj_again.project(g)
    testing.assert_close(result, result_again, equal_nan=True)


@pytest.mark.parametrize("seed, proj_type",
                         list(product([0, 1, 10**8], ['normal', 'rademacher'])))
@pytest.mark.cuda
def test_orthogonality(seed,
                       proj_type,
                       dtype=ch.float32,
                       proj_dim=1_000,
                       input_shape=(10, 10_000),
                       ):
    """
    Check that the columns of the projection matrix are orthogonal
    (we do grads @ proj_matrix)
    """
    proj = BasicProjector(grad_dim=input_shape[-1],
                          proj_dim=proj_dim,
                          proj_type=proj_type,
                          seed=seed,
                          device='cuda:0',
                          dtype=dtype
                          )

    proj_matrix = proj.proj_matrix
    num_successes  = 0
    num_trials = 300
    two_sigma = 2 * np.sqrt(proj_matrix.shape[0])
    for _ in range(num_trials):
        i, j = np.random.choice(range(proj_matrix.shape[-1]), replace=False, size=2)
        res = proj_matrix[:, i] @ proj_matrix[:, j]
        num_successes += int(res.cpu().abs().item() < two_sigma) 
    assert num_successes >= num_trials * 0.92 #  (0.95 cutting it a bit too close lol)


@pytest.mark.parametrize("seed, proj_type",
                         list(product([0, 1, 10**8], ['normal', 'rademacher'])))
@pytest.mark.cuda
def test_orthogonality_2(seed,
                         proj_type,
                         dtype=ch.float32,
                         proj_dim=1_000,
                         input_shape=(10, 10_000),
                         ):
    """
    Check that the columns of the projection matrix are orthogonal
    (we do grads @ proj_matrix)
    """
    proj = BasicProjector(grad_dim=input_shape[-1],
                          proj_dim=proj_dim,
                          proj_type=proj_type,
                          seed=seed,
                          device='cuda:0',
                          dtype=dtype
                          )

    proj_matrix = proj.proj_matrix

    proj_again = BasicProjector(grad_dim=input_shape[-1],
                                proj_dim=proj_dim,
                                proj_type=proj_type,
                                seed=seed + 10,
                                device='cuda:0',
                                dtype=dtype
                                )
    proj_matrix_again =  proj_again.proj_matrix

    num_successes  = 0
    num_trials = 300
    two_sigma = 2 * np.sqrt(proj_matrix.shape[0])
    for _ in range(num_trials):
        i = np.random.choice(range(proj_matrix.shape[-1]), size=1)
        res = proj_matrix[:, i].T @ proj_matrix_again[:, i]
        num_successes += int(res.cpu().abs().item() < two_sigma) 
    assert num_successes >= num_trials * 0.92 #  (0.95 cutting it a bit too close lol)


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
                          dtype=dtype
                          )

    # check that things break with a garbage matrix
    # (making sure the constant 15 is reasonable)
    # proj.proj_matrix = ch.empty_like(proj.proj_matrix)

    p = proj.project(g)

    delta = 0.05
    eps = np.sqrt(np.log(1 / delta) / proj_dim)
    num_trials = 100
    num_successes = 0

    for _ in range(num_trials):
        i, j = np.random.choice(range(g.shape[0]), size=2)
        n = (g[i] - g[j]).norm()
        pn = (p[i] - p[j]).norm() / proj.proj_matrix.norm(dim=1).mean()
        res = (n - pn).cpu().abs().item()
        # 15 is an arbitrary constant
        # if NaN, just give up and count as success
        num_successes += max(int(res <= 15 * eps * n), math.isinf(res))
    assert num_successes >= num_trials * (1 - 3 * delta) # leeway with 2 * 


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
                          dtype=dtype
                          )

    # check that things break with a garbage matrix
    # (making sure the constant 15 is reasonable)
    # proj.proj_matrix = ch.empty_like(proj.proj_matrix)

    p = proj.project(g)

    delta = 0.2
    eps = np.sqrt(np.log(1 / delta) / proj_dim)
    num_trials = 100
    num_successes = 0
    nrm = proj.proj_matrix.norm(dim=1).mean()

    for _ in range(num_trials):
        i, j = np.random.choice(range(g.shape[0]), size=2)
        n = (g[i] @ g[j])
        pn = ((p[i] / nrm) @ (p[j] / nrm))
        res = (n.abs() - pn.abs()).cpu().abs().item()
        t = (15 * np.sqrt(proj.proj_matrix.shape[-1]) * eps * n).abs().item()
        # if NaN, just give up and count as success
        num_successes += max(int(res <= t), math.isinf(res), math.isinf(t))
        
    assert num_successes >= num_trials * (1 - 2 * delta)


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_linearity(seed,
                   proj_type,
                   dtype,
                   proj_dim,
                   input_shape,
                   ):
    """
    Check that linearity holds (relevant for projectors that do not instantiate the
    entire JL matrix, ow trivial)
    """
    dtype = ch.float32
    g1 = testing.make_tensor(*input_shape, device='cuda:0', dtype=dtype)
    g2 = testing.make_tensor(*input_shape, device='cuda:0', dtype=dtype)

    proj = BasicProjector(grad_dim=input_shape[-1],
                          proj_dim=proj_dim,
                          proj_type=proj_type,
                          seed=seed,
                          device='cuda:0',
                          dtype=dtype
                          )

    for a in [1., 10., 45.]: # arbitrary constants
        for b in [3., 18., -24.]: # arbitrary constants
            gp = proj.project(a * g1 + b * g2)
            pg = a * proj.project(g1) + b * proj.project(g2)
            testing.assert_close(gp, pg, equal_nan=True)