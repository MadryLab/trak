#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mma.h>
#include <cuda_fp16.h>

#include <vector>

#define XSTR(x) STR(x)
#define STR(x) #x
#pragma message "The value of __CUDA__ARCH: " XSTR(__CUDA_ARCH__)

using namespace torch::indexing;
using namespace std;
using namespace nvcuda;


__global__ void fast_jl_rademacher_cuda_kernel(
        half* __restrict__ input,
        float* __restrict__ output,
        uint32_t B,
        uint32_t F,
        uint32_t N,
        uint32_t seed,
        uint32_t JL_blocks,
        uint32_t JL_tiles,
        uint32_t features_tiles,
        uint32_t feature_tile_size
) {

    /*
     * INIT_RANDOMNESS
     */
    curandStateXORWOW_t random_state;
    curand_init(
            seed,
            blockIdx.x * (32 * 32)
            + threadIdx.y * (32)
            + threadIdx.x,
            blockIdx.y * feature_tile_size, &random_state);

    /*
     * Allocate memory for data loading
     */

    __shared__ uint32_t random_bits_matrix[32][32][8];
    half *random_bits_float = reinterpret_cast<half*>(random_bits_matrix);

    wmma::fragment<wmma::accumulator, 8, 32, 16, float> accumulator;
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> data_frag;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> factors_frag;

    wmma::fill_fragment(accumulator, 0.0f);

    for (uint32_t iteration = 0 ; iteration < feature_tile_size; iteration += 16) {

        uint32_t random_bits = curand(&random_state);
        uint32_t to_write = 0;

        #pragma unroll
        for (uint32_t random_chunk = 0; random_chunk < 8; random_chunk++) {
            to_write = 1006648320 | (random_bits & 1) << 31;
            to_write |= (random_bits & 2) << 14;
            random_bits >>= 2;
            random_bits_matrix[threadIdx.y][threadIdx.x][random_chunk] = to_write;
        }

        __syncwarp();

        load_matrix_sync(data_frag,
                         input
                         + blockIdx.y * feature_tile_size
                         + iteration
                , F);

        load_matrix_sync(factors_frag,
                         random_bits_float
                         + threadIdx.y * (32 * 16),
                         16);

        wmma::mma_sync(accumulator, data_frag, factors_frag, accumulator);
    }

    wmma::store_matrix_sync(
            output
            + blockIdx.y * (N)
            + blockIdx.x * (1024)
            + threadIdx.y * (32)
            , accumulator,
            N * features_tiles,
            wmma::mem_row_major);
}

void fast_jl_rademacher_cuda(
        torch::Tensor input,
        torch::Tensor output,
        uint32_t seed,
        uint32_t num_batches,
        uint32_t JL_blocks,
        uint32_t JL_tiles,
        uint32_t features_tiles,
        uint32_t feature_tile_size
) {

    for (uint32_t batch = 0; batch < num_batches; batch++) {
        auto batch_start = batch * 8;
        auto batch_end = (batch + 1) * 8;
        auto batch_slice = Slice(batch_start, batch_end, None);
        auto current_input = input.index({batch_slice});
        auto current_output = output.index({batch_slice});

        uint32_t B = input.size(0);
        uint32_t F = input.size(1);
        uint32_t N = output.size(2);

        dim3 gridDim(JL_tiles, features_tiles);
        dim3 blockDim(32, 32);

        fast_jl_rademacher_cuda_kernel<<<gridDim, blockDim>>>(
                (__half*) current_input.data_ptr<at::Half>(),
                current_output.data_ptr<float>(),
                B,
                F,
                N,
                seed,
                JL_blocks,
                JL_tiles,
                features_tiles,
                feature_tile_size);
    }
}