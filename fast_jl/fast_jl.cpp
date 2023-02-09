#include <torch/extension.h>

#include <vector>

void fast_jl_rademacher_cuda(
    torch::Tensor input,
    uint32_t seed,
    torch::Tensor output);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void fast_jl_rademacher(
    torch::Tensor input,
    uint32_t seed,
    torch::Tensor output) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  return fast_jl_rademacher_cuda(input, seed, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rademacher", &fast_jl_rademacher, "Fast Rademacher Projection (CUDA)");
}
