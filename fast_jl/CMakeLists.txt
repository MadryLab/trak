cmake_minimum_required(VERSION 3.24)
project(fast_jl)

set(CMAKE_CUDA_STANDARD 14)

include_directories( /home/guillaumeleclerc/miniconda3/envs/fastjl/lib/python3.10/site-packages/torch/include/torch/csrc/api/include)
include_directories( /home/guillaumeleclerc/miniconda3/envs/fastjl/lib/python3.10/site-packages/torch/include/)

add_executable(fast_jl
        fast_jl.cu
        fast_jl_kernel.cuh data_loading.cuh)