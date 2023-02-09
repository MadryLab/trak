import pytest
from itertools import product

import torch as ch
from torch import testing

try:
    import fast_jl
except:
    print(f'No fast_jl available!')
from assertpy import assert_that

PARAM = list(product([8], [1024, 2048], [5], [0, 1]))

@pytest.mark.parametrize("bs, input_size, output_size, seed ", PARAM)
@pytest.mark.cuda
def test_shape(bs: int, input_size: int, output_size: int, seed:int):
    input_data = ch.ones((bs, input_size), dtype=ch.float16, device="cuda:0")

    result = fast_jl.rademacher(input_data, output_size, seed)
    assert_that(result.shape).is_equal_to((bs, output_size))

@pytest.mark.cuda
def test_running():
    bs = 8
    input_size = 256
    seed = 17
    output_size = 32
    input_data = ch.ones((bs, input_size), dtype=ch.float16, device="cuda:0")

    result = fast_jl.rademacher(input_data, output_size, seed)
    print(result)
    print(result.sum())

@pytest.mark.cuda
def test_even():
    bs = 8
    input_size = 10240
    seed = 64
    output_size = 64
    input_data = ch.ones((bs, input_size), dtype=ch.float16, device="cuda:0")

    result = fast_jl.rademacher(input_data, output_size, seed)
    assert_that(ch.all(result % 2 == 0)).is_true()

@pytest.mark.cuda
def test_odd():
    bs = 8
    input_size = 10241
    seed = 78
    output_size = 64
    input_data = ch.ones((bs, input_size), dtype=ch.float16, device="cuda:0")

    result = fast_jl.rademacher(input_data, output_size, seed)
    assert_that(ch.all(result % 2 == 1)).is_true()

