#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mma.h>
#include <cuda_fp16.h>

#include <vector>

namespace {
    __global__ void fast_jl_rademacher_cuda_kernel(
            float* __restrict__ output) {
        curandStateXORWOW_t random_state;
        curand_init(0, 0, 5, &random_state);
        output[0] = (float) curand(&random_state);
    }
} // namespace

void fast_jl_rademacher_cuda(
        torch::Tensor input,
        uint32_t seed,
        torch::Tensor output) {
    fast_jl_rademacher_cuda_kernel<<<2, 2>>>(output.data<float>());
}