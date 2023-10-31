import pytest
from itertools import product

import torch as ch

try:
    import fast_jl
except ModuleNotFoundError:
    print("No fast_jl available!")

from assertpy import assert_that

bs_error_str = "CUDA error: too many resources requested for launch\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1.\nCompile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n"  # noqa

new_bs_error_str = f"The batch size of the CudaProjector is too large for your GPU. Reduce it by using the max_batch_size argument of the CudaProjector.\nOriginal error: {bs_error_str}"  # noqa


PARAM = list(product([8], [1024, 2048], [512, 1024, 2048], [0, 1]))


@pytest.mark.parametrize("bs, input_size, output_size, seed ", PARAM)
@pytest.mark.cuda
def test_shape(bs: int, input_size: int, output_size: int, seed: int):
    print(output_size)
    input_data = ch.ones((bs, input_size), dtype=ch.float16, device="cuda:0")

    num_sms = ch.cuda.get_device_properties(
        ch.cuda.current_device()
    ).multi_processor_count

    try:
        result = fast_jl.project_rademacher_8(input_data, output_size, seed, num_sms)
    except RuntimeError as e:
        if str(e) == bs_error_str:
            raise RuntimeError(new_bs_error_str)
        else:
            raise e

    assert_that(result.shape).is_equal_to((bs, output_size))


@pytest.mark.cuda
def test_running():
    bs = 8
    input_size = 256
    seed = 17
    output_size = 512
    input_data = ch.ones((bs, input_size), dtype=ch.float16, device="cuda:0")

    num_sms = ch.cuda.get_device_properties(
        ch.cuda.current_device()
    ).multi_processor_count

    try:
        result = fast_jl.project_rademacher_8(input_data, output_size, seed, num_sms)
    except RuntimeError as e:
        if str(e) == bs_error_str:
            raise RuntimeError(new_bs_error_str)
        else:
            raise e

    print(result.sum())


@pytest.mark.cuda
def test_even():
    bs = 8
    input_size = 10240
    seed = 64
    output_size = 1024
    input_data = ch.ones((bs, input_size), dtype=ch.float16, device="cuda:0")

    num_sms = ch.cuda.get_device_properties(
        ch.cuda.current_device()
    ).multi_processor_count

    try:
        result = fast_jl.project_rademacher_8(input_data, output_size, seed, num_sms)
    except RuntimeError as e:
        if str(e) == bs_error_str:
            raise RuntimeError(new_bs_error_str)
        else:
            raise e

    assert_that(ch.all(result % 2 == 0)).is_true()


@pytest.mark.cuda
def test_odd():
    bs = 8
    input_size = 10241
    seed = 78
    output_size = 2048
    input_data = ch.ones((bs, input_size), dtype=ch.float16, device="cuda:0")

    num_sms = ch.cuda.get_device_properties(
        ch.cuda.current_device()
    ).multi_processor_count

    try:
        result = fast_jl.project_rademacher_8(input_data, output_size, seed, num_sms)
    except RuntimeError as e:
        if str(e) == bs_error_str:
            raise RuntimeError(new_bs_error_str)
        else:
            raise e

    assert_that(ch.all(result % 2 == 1)).is_true()
