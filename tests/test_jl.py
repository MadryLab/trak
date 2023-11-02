import pytest
import math
from itertools import product
import numpy as np
import torch
from torch import testing

from trak.projectors import CudaProjector, ProjectionType, ChunkedCudaProjector

ch = torch


def get_max_chunk_size(
    batch_size: int,
) -> tuple[int, list]:
    max_chunk_size = np.iinfo(np.uint32).max // batch_size
    return max_chunk_size


def make_input(
    input_shape, max_chunk_size, device="cuda", dtype=torch.float32, g_tensor=None
):
    if g_tensor is None:
        g = testing.make_tensor(*input_shape, device=device, dtype=dtype)
    else:
        g = g_tensor
    _, num_params = input_shape
    num_chunks = np.ceil(num_params / max_chunk_size).astype("int32")
    g_chunks = ch.chunk(g, num_chunks, dim=1)
    result = {}
    for i, x in enumerate(g_chunks):
        result[i] = x
        print(f"Input param group {i} shape: {x.shape}")

    return result


BasicProjector = CudaProjector

MAX_BATCH_SIZE = 32
PARAM = list(
    product(
        [0, 1, 10**8],  # seed
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
    )
)

PARAM = list(
    product(
        [123],  # seed
        [ProjectionType.rademacher],  # proj type
        [ch.float32],  # dtype
        [
            # tests for MAXINT32 overflow
            (8, 180645096),  # pass: np.prod(shape) < np.iinfo(np.int32).max
            (31, 180645096),  # fail: np.prod(shape) > np.iinfo(np.int32).max
            (32, 180645096),  # fail: np.prod(shape) > np.iinfo(np.int32).max
            (2, 780645096),  # fail: np.prod(shape) > np.iinfo(np.int32).max
        ],  # input shape
        [15_360],  # proj dim
    )
)


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_seed_consistency(
    seed,
    proj_type,
    dtype,
    proj_dim,
    input_shape,
):
    """
    Check that re-running the same projection with the same seed
    leads to the same result.
    """

    proj = BasicProjector(
        grad_dim=input_shape[-1],
        proj_dim=proj_dim,
        proj_type=proj_type,
        seed=seed,
        device="cuda:0",
        dtype=dtype,
        max_batch_size=MAX_BATCH_SIZE,
    )
    batch_size = input_shape[0]
    max_chunk_size = get_max_chunk_size(batch_size)
    g = make_input(input_shape, max_chunk_size, "cuda:0", dtype)

    result = proj.project(g, model_id=0)
    result_again = proj.project(g, model_id=0)
    testing.assert_close(result, result_again, equal_nan=True)

    del g
    torch.cuda.empty_cache()


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_seed_consistency_2(
    seed,
    proj_type,
    dtype,
    proj_dim,
    input_shape,
):
    """
    Check that re-initializing the class and re-running the same projection
    with the same seed leads to the same result.
    """

    batch_size = input_shape[0]
    max_chunk_size = get_max_chunk_size(batch_size)
    g = make_input(input_shape, max_chunk_size, "cuda:0", dtype)

    proj = BasicProjector(
        grad_dim=input_shape[-1],
        proj_dim=proj_dim,
        proj_type=proj_type,
        seed=seed,
        device="cuda:0",
        dtype=dtype,
        max_batch_size=MAX_BATCH_SIZE,
    )

    result = proj.project(g, model_id=0)

    proj_again = BasicProjector(
        grad_dim=input_shape[-1],
        proj_dim=proj_dim,
        proj_type=proj_type,
        seed=seed,
        device="cuda:0",
        dtype=dtype,
        max_batch_size=MAX_BATCH_SIZE,
    )
    result_again = proj_again.project(g, model_id=0)
    testing.assert_close(result, result_again, equal_nan=True)

    del g
    torch.cuda.empty_cache()


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_norm_preservation(
    seed,
    proj_type,
    dtype,
    proj_dim,
    input_shape,
):
    """
    Check that norms of differences are approximately preserved.
    """
    batch_size = input_shape[0]
    max_chunk_size = get_max_chunk_size(batch_size)
    g = make_input(input_shape, max_chunk_size, "cuda:0", dtype)

    rng = np.random.default_rng(seed)
    seeds = rng.integers(
        low=0,
        high=500,
        size=len(g),
    )

    param_chunk_sizes = [v.size(1) for v in g.values()]
    projector_per_chunk = [
        BasicProjector(
            grad_dim=chunk_size,
            proj_dim=proj_dim,
            seed=seeds[i],
            proj_type=proj_type,
            max_batch_size=MAX_BATCH_SIZE,
            dtype=dtype,
            device="cuda:0",
        )
        for i, chunk_size in enumerate(param_chunk_sizes)
    ]
    proj = ChunkedCudaProjector(
        projector_per_chunk,
        max_chunk_size,
        param_chunk_sizes,
        batch_size,
        "cuda:0",
        dtype,
    )

    p = proj.project(g, model_id=0)

    delta = 0.05
    eps = np.sqrt(np.log(1 / delta) / proj_dim)
    num_trials = 100
    num_successes = 0

    # flatten
    g = ch.cat([v for v in g.values()], dim=1)
    print(f"Flattened input shape: {g.shape}")

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
            print("aaaaaa")
        num_successes += int(res <= 35 * eps * n)
    assert num_successes >= num_trials * (1 - 3 * delta)  # leeway with 2 *

    del g
    torch.cuda.empty_cache()


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_prod_preservation(
    seed,
    proj_type,
    dtype,
    proj_dim,
    input_shape,
):
    """
    Check that dot products are approximately preserved.
    """
    batch_size = input_shape[0]
    max_chunk_size = get_max_chunk_size(batch_size)
    g = make_input(input_shape, max_chunk_size, "cuda:0", dtype)

    proj = BasicProjector(
        grad_dim=input_shape[-1],
        proj_dim=proj_dim,
        proj_type=proj_type,
        seed=seed,
        device="cuda:0",
        dtype=dtype,
        max_batch_size=MAX_BATCH_SIZE,
    )

    # check that things break with a garbage matrix
    # (making sure the constant 15 is reasonable)
    # proj.proj_matrix = ch.empty_like(proj.proj_matrix)

    p = proj.project(g, model_id=0)

    delta = 0.2
    eps = np.sqrt(np.log(1 / delta) / proj_dim)
    num_trials = 100
    num_successes = 0

    # flatten
    g = ch.cat([v for v in g.values()], dim=1)
    print(f"Flattened input shape: {g.shape}")

    for _ in range(num_trials):
        i, j = np.random.choice(range(g.shape[0]), size=2)
        n = g[i] @ g[j]
        pn = (p[i] / np.sqrt(input_shape[-1])) @ (p[j] / input_shape[-1])
        res = (n.abs() - pn.abs()).cpu().abs().item()
        t = (50 * np.sqrt(proj_dim) * eps * n).abs().item()
        # if NaN, just give up and count as success
        num_successes += max(int(res <= t), math.isinf(res), math.isinf(t))

    assert num_successes >= num_trials * (1 - 2 * delta)

    del g
    torch.cuda.empty_cache()


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_single_nonzero_feature(
    seed,
    proj_type,
    dtype,
    proj_dim,
    input_shape,
):
    """
    Check that output takes into account every feature.
    """

    batch_size = input_shape[0]
    max_chunk_size = get_max_chunk_size(batch_size)
    g = make_input(input_shape, max_chunk_size, "cuda:0", dtype)
    for k in g.keys():
        g[k] = ch.zeros_like(g[k])

    for ind in range(input_shape[0]):
        param_group = np.random.choice(range(len(g.keys())))
        coord = np.random.choice(range(g[param_group].size(1)))
        val = ch.randn(1)
        g[param_group][ind, coord] = val.item()

    proj = BasicProjector(
        grad_dim=input_shape[-1],
        proj_dim=proj_dim,
        proj_type=proj_type,
        seed=seed,
        device="cuda:0",
        dtype=dtype,
        max_batch_size=MAX_BATCH_SIZE,
    )
    p = proj.project(g, model_id=0)
    assert (~ch.isclose(p, ch.zeros_like(p))).all().item()


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_first_nonzero_feature(
    seed,
    proj_type,
    dtype,
    proj_dim,
    input_shape,
):
    """
    Check that output takes into account first features.
    """
    g = ch.zeros(*input_shape, device="cuda:0", dtype=dtype)
    g[:, 0] = 1.0

    batch_size = input_shape[0]
    max_chunk_size = get_max_chunk_size(batch_size)
    g = make_input(input_shape, max_chunk_size, g_tensor=g)
    print(g[0])

    proj = BasicProjector(
        grad_dim=input_shape[-1],
        proj_dim=proj_dim,
        proj_type=proj_type,
        seed=seed,
        device="cuda:0",
        dtype=dtype,
        max_batch_size=MAX_BATCH_SIZE,
    )
    p = proj.project(g, model_id=0)
    assert (~ch.isclose(p, ch.zeros_like(p))).all().item()


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_last_nonzero_feature(
    seed,
    proj_type,
    dtype,
    proj_dim,
    input_shape,
):
    """
    Check that output takes into account last features.
    """
    g = ch.zeros(*input_shape, device="cuda:0", dtype=dtype)
    g[:, -1] = 1.0

    batch_size = input_shape[0]
    max_chunk_size = get_max_chunk_size(batch_size)
    g = make_input(input_shape, max_chunk_size, g_tensor=g)
    print(g[0])

    proj = BasicProjector(
        grad_dim=input_shape[-1],
        proj_dim=proj_dim,
        proj_type=proj_type,
        seed=seed,
        device="cuda:0",
        dtype=dtype,
        max_batch_size=MAX_BATCH_SIZE,
    )
    p = proj.project(g, model_id=0)
    assert (~ch.isclose(p, ch.zeros_like(p))).all().item()


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_same_features(
    seed,
    proj_type,
    dtype,
    proj_dim,
    input_shape,
):
    """
    Check that output is the same for the same features
    """
    g = testing.make_tensor(*input_shape, device="cuda:0", dtype=dtype)
    g[-1] = g[0]

    batch_size = input_shape[0]
    max_chunk_size = get_max_chunk_size(batch_size)
    g = make_input(input_shape, max_chunk_size, g_tensor=g)
    for i in range(len(g)):
        print(g[i][0] == g[i][-1])

    rng = np.random.default_rng(seed)
    seeds = rng.integers(
        low=0,
        high=500,
        size=len(g),
    )

    param_chunk_sizes = [v.size(1) for v in g.values()]
    projector_per_chunk = [
        BasicProjector(
            grad_dim=chunk_size,
            proj_dim=proj_dim,
            seed=seeds[i],
            proj_type=proj_type,
            max_batch_size=MAX_BATCH_SIZE,
            dtype=dtype,
            device="cuda:0",
        )
        for i, chunk_size in enumerate(param_chunk_sizes)
    ]
    proj = ChunkedCudaProjector(
        projector_per_chunk,
        max_chunk_size,
        param_chunk_sizes,
        batch_size,
        "cuda:0",
        dtype,
    )
    p = proj.project(g, model_id=0)

    assert ch.allclose(p[0], p[-1])

    del g
    torch.cuda.empty_cache()


@pytest.mark.parametrize("seed, proj_type, dtype, input_shape, proj_dim", PARAM)
@pytest.mark.cuda
def test_orthogonality(
    seed,
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
        proj = BasicProjector(
            grad_dim=input_shape[-1],
            proj_dim=proj_dim,
            proj_type=proj_type,
            seed=seed,
            device="cuda:0",
            dtype=dtype,
            max_batch_size=MAX_BATCH_SIZE,
        )

        num_successes = 0
        num_trials = 10
        for _ in range(num_trials):
            g = testing.make_tensor(*input_shape, device="cuda:0", dtype=dtype)
            g[-1] -= g[0] @ g[-1] / (g[0].norm() ** 2) * g[0]

            batch_size = input_shape[0]
            max_chunk_size = get_max_chunk_size(batch_size)
            g = make_input(input_shape, max_chunk_size, g_tensor=g)

            p = proj.project(g, model_id=0)
            if p[0] @ p[-1] < 1e-3:
                num_successes += 1
        assert num_successes > 0.33 * num_trials

    del g
    torch.cuda.empty_cache()
