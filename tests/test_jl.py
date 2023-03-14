import pytest
import math
from itertools import product
import numpy as np
import torch as ch
from torch import testing

from trak.projectors import CudaProjector, ProjectionType
BasicProjector = CudaProjector

PARAM = list(product([0, 1, 10**8],  # seed
                     [ProjectionType.normal, ProjectionType.rademacher],  # proj type
                     [ch.float16, ch.float32],  # dtype
                     [
                         (1, 25),
                         (8, 10_000),
                         (16, 10_002),
                         (9, 10_002),
                         (16, 10_001),
                         (45, 1049),
                         (1, int(1e9)),
                     ],  # input shape
                     [4096, 1024],  # proj dim
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
                          dtype=dtype
                          )

    result = proj.project(g, model_id=0)

    proj_again = BasicProjector(grad_dim=input_shape[-1],
                                proj_dim=proj_dim,
                                proj_type=proj_type,
                                seed=seed,
                                device='cuda:0',
                                dtype=dtype)
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
                          dtype=dtype
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
                          dtype=dtype
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
                          dtype=dtype
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
                          dtype=dtype
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
                          dtype=dtype
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
    g[-1] = g[0]

    proj = BasicProjector(grad_dim=input_shape[-1],
                          proj_dim=proj_dim,
                          proj_type=proj_type,
                          seed=seed,
                          device='cuda:0',
                          dtype=dtype
                          )
    p = proj.project(g, model_id=0)

    assert ch.allclose(p[0], p[-1])


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
                              dtype=dtype)

        num_successes = 0
        num_trials = 100
        for _ in range(num_trials):
            g = testing.make_tensor(*input_shape, device='cuda:0', dtype=dtype)
            g[-1] -= g[0] @ g[-1] / (g[0].norm() ** 2) * g[0]
            p = proj.project(g, model_id=0)
            if p[0] @ p[-1] < 1e-3:
                num_successes += 1
        assert num_successes > 0.33 * num_trials
