#include <vector>
#include <algorithm>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <cuda_fp16.h>

#include "types.cuh"
#include "fast_jl_kernel.cuh"

using namespace torch::indexing;

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_2D(x) AT_ASSERTM(x.dim() == 2, #x " must be 2D")
#define CHECK_HALF(x) AT_ASSERTM(x.dtype() == torch::kFloat16 || x.dtype() == torch::kFloat32, #x " must be fp16 or fp32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_2D(x); CHECK_HALF(x);

template<ProjectionType p_type, uint32_t NUM_BATCHES>
torch::Tensor fast_jl(
        torch::Tensor input,
        uint32_t N,
        uint32_t seed,
        uint32_t num_feature_tiles
) {

    CHECK_INPUT(input);

    uint32_t B = input.size(0);
    uint32_t F = input.size(1);

    num_feature_tiles = min(F, num_feature_tiles);

    uint32_t effective_batch_size = 8 * NUM_BATCHES;
    uint32_t num_required_batches = (B - 1) / effective_batch_size + 1;
    uint32_t num_batch_dim = num_required_batches * effective_batch_size;

    auto output = torch::zeros({num_batch_dim, num_feature_tiles, N},
                               torch::TensorOptions().device(input.device()));

    for (uint32_t meta_batch=0; meta_batch < num_required_batches; meta_batch++) {
        uint32_t batch_start = meta_batch * effective_batch_size;
        uint32_t batch_end = (meta_batch + 1) * effective_batch_size;

        auto batch_slice = Slice(batch_start, batch_end, None);
        auto current_input = input.index({batch_slice});
        auto current_output = output.index({batch_slice});

        uint32_t real_batch_end = std::min(batch_end, B) % effective_batch_size;
        if (real_batch_end == 0) real_batch_end = effective_batch_size;

        if (input.dtype() == torch::kFloat16) {
            project<__half, p_type, NUM_BATCHES, 16>(
                    (__half*) current_input.data_ptr<at::Half>(),
                    current_output.data_ptr<float>(),
                    real_batch_end, F, N,
                    seed, num_feature_tiles);
        } else {
            project<float, p_type, NUM_BATCHES, 16>(
                    current_input.data_ptr<float>(),
                    current_output.data_ptr<float>(),
                    real_batch_end, F, N,
                    seed, num_feature_tiles);

        }
    }

    return output.index({Slice({0, B})}).sum(1);
}

torch::Tensor proj_rademacher_8(torch::Tensor input, uint32_t N, uint32_t seed, uint32_t num_feature_tiles) {
    return fast_jl<Rademacher, 1>(input, N, seed, num_feature_tiles);
}
torch::Tensor proj_rademacher_16(torch::Tensor input, uint32_t N, uint32_t seed, uint32_t num_feature_tiles) {
    return fast_jl<Rademacher, 2>(input, N, seed, num_feature_tiles);
}
torch::Tensor proj_rademacher_32(torch::Tensor input, uint32_t N, uint32_t seed, uint32_t num_feature_tiles) {
    return fast_jl<Rademacher, 4>(input, N, seed, num_feature_tiles);
}
torch::Tensor proj_normal_8(torch::Tensor input, uint32_t N, uint32_t seed, uint32_t num_feature_tiles) {
    return fast_jl<Normal, 1>(input, N, seed, num_feature_tiles);
}
torch::Tensor proj_normal_16(torch::Tensor input, uint32_t N, uint32_t seed, uint32_t num_feature_tiles) {
    return fast_jl<Normal, 2>(input, N, seed, num_feature_tiles);
}
torch::Tensor proj_normal_32(torch::Tensor input, uint32_t N, uint32_t seed, uint32_t num_feature_tiles) {
    return fast_jl<Normal, 4>(input, N, seed, num_feature_tiles);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project_rademacher_8", &proj_rademacher_8, "Fast Random Projection (CUDA)");
    m.def("project_rademacher_16", &proj_rademacher_16, "Fast Random Projection (CUDA)");
    m.def("project_rademacher_32", &proj_rademacher_32, "Fast Random Projection (CUDA)");

    m.def("project_normal_8", &proj_normal_8, "Fast Random Projection (CUDA)");
    m.def("project_normal_16", &proj_normal_16, "Fast Random Projection (CUDA)");
    m.def("project_normal_32", &proj_normal_32, "Fast Random Projection (CUDA)");
}
