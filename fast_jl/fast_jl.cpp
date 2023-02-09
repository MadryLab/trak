#include <torch/extension.h>

#define NUM_SMS 108

#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace torch::indexing;

void fast_jl_rademacher_cuda(
        torch::Tensor input,
        torch::Tensor output,
        uint32_t seed,
        uint32_t num_batches,
        uint32_t JL_blocks,
        uint32_t JL_tiles,
        uint32_t features_tiles,
        uint32_t feature_tile_size
);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_BIG(x) AT_ASSERTM(x.size(1) >= 1024, #x " must have at least 1024 components")
#define CHECK_2D(x) AT_ASSERTM(x.dim() == 2, #x " must be 2D")
#define CHECK_HALF(x) AT_ASSERTM(x.dtype() == torch::kFloat16, #x " must be Float16")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_2D(x); CHECK_HALF(x);

torch::Tensor fast_jl_rademacher(
        torch::Tensor input,
        uint32_t N,
        uint32_t seed
        ) {

    CHECK_INPUT(input);

    uint32_t B = input.size(0);
    uint32_t F = input.size(1);


    uint32_t num_batches = (B - 1) / 8 + 1;
    uint32_t JL_blocks = (N - 1) / 32 + 1;
    uint32_t JL_tiles = (N - 1) / 1024 + 1;
    uint32_t features_tiles = (NUM_SMS - 1) / JL_tiles + 1;
    uint32_t feature_tile_size = F / features_tiles;
    feature_tile_size -= feature_tile_size % 16;
    features_tiles = F / feature_tile_size;
    uint32_t remaining = F - feature_tile_size * features_tiles;

    std::cout << "Remaining: " << remaining << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "ft: " << features_tiles << std::endl;
    std::cout << "fts: " << feature_tile_size << std::endl;
    std::cout << "jlb: " << JL_blocks << std::endl;
    std::cout << "jlt: " << JL_tiles << std::endl;

    auto result = torch::zeros({B, features_tiles, N},
                               torch::TensorOptions().device(input.device()));

    fast_jl_rademacher_cuda(
            input,
            result,
            seed,
            num_batches,
            JL_blocks,
            JL_tiles,
            features_tiles,
            feature_tile_size);

    auto partial_sum = result.sum(1);

    auto generator = at::cuda::detail::createCUDAGenerator();
    generator.set_current_seed(seed);
    auto leftover_jl = torch::rand({remaining, N}, generator, torch::TensorOptions().device(input.device()));
    leftover_jl = torch::round(leftover_jl) * 2 - 1;

    return partial_sum + torch::matmul(input.index({
        Ellipsis,
        Slice(input.size(1) - remaining, None, None)}).to(torch::kFloat), leftover_jl);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rademacher", &fast_jl_rademacher, "Fast Rademacher Projection (CUDA)");
}
