#include <iostream>
#include <type_traits>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include "random.cuh"
#include "error_handling.cuh"
#include "types.cuh"

#define WARP_SIZE 32
#define CHUNK_COL 32
#define CHUNK_ROW 8

using namespace std;
using namespace nvcuda::wmma;

template<typename InputType, uint32_t CHUNKS_PER_TILE>
__device__
void load_into_shared_memory(
        const InputType* __restrict__ input,
        half input_buffer[CHUNKS_PER_TILE][CHUNK_ROW][CHUNK_COL],
        uint32_t channels,
        uint32_t features,
        uint32_t feature_tile_size,
        uint32_t iteration,
        uint32_t batch) {

    uint32_t current_col = (
            blockIdx.y * feature_tile_size
            + threadIdx.z * CHUNK_COL
            + threadIdx.x
            + iteration);

    uint32_t current_row = (
            batch * CHUNK_ROW
            + threadIdx.y);

    for (uint32_t iter=0; iter < CHUNK_ROW; iter++) {
        half value;

        if (current_col >= features || current_row >= channels) {
            value = __float2half(0.0f);
        } else {
            const InputType *my_input = (input
                                         + current_row * features
                                         + current_col
            );
            if (is_same<InputType, float>::value) {
                value = __float2half(*my_input);
            } else {
                value = *my_input;
            }
        }
        input_buffer[threadIdx.z][threadIdx.y + blockDim.y * iter][threadIdx.x] = value;
        current_col += blockDim.y;
    }
}

template<typename InputType, ProjectionType p_type, uint32_t NUM_BATCHES, uint32_t CHUNKS_PER_TILE>
__global__ void
project_kernel(const float *__restrict__ input,
               float *__restrict__ output,
               uint32_t channels,
               uint32_t features,
               uint32_t output_dims,
               uint32_t seed,
               uint32_t feature_tile_size) {

    // Which column(=JL Dim) of the output this thread is responsible for
    uint32_t col_output = blockIdx.x    * (gridDim.z * gridDim.y * gridDim.x)
                          + threadIdx.z * (            gridDim.y * gridDim.x)
                          + threadIdx.y * (                        gridDim.x)
                          + threadIdx.x;

    // Init Random State
    curandStateXORWOW_t random_state;
    jl_random::init(random_state, col_output, blockIdx.y, seed);

    __shared__ half input_buffer[NUM_BATCHES][CHUNKS_PER_TILE][CHUNK_ROW][CHUNK_COL];
    __shared__ half factors[CHUNKS_PER_TILE][CHUNK_ROW][CHUNK_COL];
    // Pointer to the location where this warp will store its random coefficients
    half* warp_factors = &factors[threadIdx.z][0][0];

    fragment<matrix_a, CHUNK_ROW, CHUNK_COL, 16, half, row_major> data_fragment;
    fragment<matrix_b, CHUNK_ROW, CHUNK_COL, 16, half, row_major> factors_fragment;
    fragment<accumulator, CHUNK_ROW, CHUNK_COL, 16, float> accumulator[NUM_BATCHES];

    for (uint32_t batch = 0; batch < NUM_BATCHES; batch++) {
        fill_fragment((accumulator[batch]), 0.0f);
    }

    for (uint32_t iteration = 0; iteration < feature_tile_size; iteration += CHUNK_COL * CHUNKS_PER_TILE) {
        // We load all the data for all the batches and all the chunks
        for (uint32_t batch = 0 ; batch < NUM_BATCHES; batch++) {
            load_into_shared_memory<InputType, CHUNKS_PER_TILE>(
                    input,
                    input_buffer[batch],
                    channels,
                    features,
                    feature_tile_size,
                    iteration,
                    batch);
        }

        __syncthreads(); // Make sure the data has been read before proceeding with the computation

#pragma unroll
        for (uint32_t cur_chunk = 0; cur_chunk < CHUNKS_PER_TILE; cur_chunk++) {

            // Generate and load the random coefficients (These are shared for all the batches)
            jl_random::generate_factors_fragment<p_type>(warp_factors, random_state);
            load_matrix_sync(factors_fragment, warp_factors , CHUNK_COL);
#pragma unroll
            for (uint32_t batch = 0 ; batch < NUM_BATCHES; batch++) {
                load_matrix_sync(data_fragment, &input_buffer[batch][cur_chunk][0][0], 16);
                mma_sync(accumulator[batch], data_fragment, factors_fragment, accumulator[batch]);
            }
        }

        // Let's not start overwrite stuff while some threads might still be using it
        __syncthreads();
    }

    for (uint32_t batch = 0 ; batch < NUM_BATCHES; batch++) {
        store_matrix_sync(
                output
                + batch * (CHUNK_ROW * output_dims)
                + col_output,
                accumulator[batch], CHUNK_COL, mem_row_major);
    }
}


template<typename InputType, ProjectionType p_type, uint32_t NUM_BATCHES, uint32_t CHUNKS_PER_TILE>
float *project(const float *__restrict__ input,
               uint32_t channels, uint32_t features, uint32_t output_dims,
               uint32_t seed, uint32_t num_SMs) {

    if (channels == 0 || channels > CHUNK_ROW * NUM_BATCHES) {
        throw invalid_argument("Invalid number of channels (has to be in [1, 8 * NUM_BATCHES])");
    }

    uint32_t num_chunks = (output_dims - 1) / CHUNK_COL + 1;
    uint32_t num_jl_tiles = (num_chunks - 1) / CHUNKS_PER_TILE + 1;

    uint32_t num_feature_tiles = num_SMs * 1;
    uint32_t feature_tile_size = (features - 1) / (num_feature_tiles) + 1;

    dim3 blockSize(CHUNK_COL, WARP_SIZE / CHUNK_COL, CHUNKS_PER_TILE);
    dim3 gridSize(num_jl_tiles, num_feature_tiles , 1);

    float *output;
    cudaMalloc(&output, channels * num_feature_tiles * output_dims * sizeof(float));

    cout << blockSize.x << ", " << blockSize.y << ", " << blockSize.z << std::endl;
    cout << gridSize.x << ", " << gridSize.y << ", " << gridSize.z << std::endl;
    cout << "FTS " << feature_tile_size << endl;

    project_kernel<InputType, p_type, NUM_BATCHES, CHUNKS_PER_TILE>
            <<<gridSize, blockSize>>>
            (input, output, CHUNK_ROW * NUM_BATCHES, features, output_dims, seed, feature_tile_size);

    std::cout <<"HOHO" << std::endl;
    return output;
}

int main() {

    uint32_t C = 32;
    uint32_t F = 10485760;
    uint32_t N = 512;

    float *input;

    cudaMalloc(&input, C * F * sizeof(float));

    cudaDeviceSynchronize();

    float* output;
    for (int i = 0; i < 1; i++) {
        output = project<float, ProjectionType::Rademacher, 4, 16>(input, C, F, N, 0, 56);
    }

    cudaDeviceSynchronize();

    cudaFree(input);
    cudaFree(output);

    CHECK_LAST_CUDA_ERROR();
    return 0;
}
