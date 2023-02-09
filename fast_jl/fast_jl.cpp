#include <torch/extension.h>

#define NUM_SMS 108

#include <vector>

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
#define CHECK_2D(x) AT_ASSERTM(x.dim() == 2, #x " must be 2D")
#define CHECK_HALF(x) AT_ASSERTM(x.dtype() == torch::kFloat16, #x " must be Float16")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_2D(x); CHECK_HALF(x)

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
    uint32_t feature_tile_size = (F - 1) / features_tiles + 1;

    auto result = torch::empty({B, features_tiles, N},
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

    return result.sum(1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rademacher", &fast_jl_rademacher, "Fast Rademacher Projection (CUDA)");
}
