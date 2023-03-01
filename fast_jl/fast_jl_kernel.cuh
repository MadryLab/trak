#include <iostream>
#include <string>
#include <type_traits>
#include <cuda_fp16.h>
#include <vector_types.h>
#include <curand_kernel.h>
#include <cuda_pipeline.h>
#include <mma.h>

#define WARP_SIZE 32
#define CHUNK_COL 32
#define CHUNK_ROW 8

#include "random.cuh"
#include "error_handling.cuh"
#include "data_loading.cuh"
#include "types.cuh"


using namespace std;
using namespace nvcuda::wmma;


template<typename InputType, ProjectionType p_type, uint32_t NUM_BATCHES, uint32_t CHUNKS_PER_TILE>
__global__ void
project_kernel(InputType *__restrict__ input,
               float *__restrict__ output,
               uint32_t channels,
               uint32_t features,
               uint32_t output_dims,
               uint32_t seed,
               uint32_t feature_tile_size) {

    uint32_t lane_id = threadIdx.x + blockDim.x * threadIdx.y;

    // Which column(=JL Dim) of the output this thread is responsible for
    uint32_t col_output = (
            lane_id
            + threadIdx.z * (WARP_SIZE)
            + blockIdx.x  * (WARP_SIZE * blockDim.z)
    );

    // Init Random State
    curandStateXORWOW_t random_state;
    jl_random::init(random_state, col_output, blockIdx.y, output_dims, seed);

    __shared__ half input_buffer[NUM_BATCHES][CHUNKS_PER_TILE][CHUNK_ROW][16];
    __shared__ half factors[CHUNKS_PER_TILE][16][CHUNK_COL];
    // Pointer to the location where this warp will store its random coefficients
    half* warp_factors = &factors[threadIdx.z][0][0];

    fragment<matrix_a, CHUNK_ROW, CHUNK_COL, 16, half, row_major> data_fragment;
    fragment<matrix_b, CHUNK_ROW, CHUNK_COL, 16, half, row_major> factors_fragment;
    fragment<accumulator, CHUNK_ROW, CHUNK_COL, 16, float> accumulator[NUM_BATCHES];

    for (uint32_t batch = 0; batch < NUM_BATCHES; batch++) {
        fill_fragment((accumulator[batch]), 0.0f);
    }

    for (uint32_t iteration = 0; iteration < feature_tile_size; iteration += 16 * CHUNKS_PER_TILE) {
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
        uint32_t col_output_warp = (
                threadIdx.z * WARP_SIZE
                + blockIdx.x * (WARP_SIZE * blockDim.z)
                + blockIdx.y * output_dims);

        store_matrix_sync(
                output
                + batch * (output_dims * gridDim.y * CHUNK_ROW)
                + col_output_warp,
                accumulator[batch], output_dims * gridDim.y, mem_row_major);
    }
}

template<typename InputType, ProjectionType p_type, uint32_t NUM_BATCHES, uint32_t CHUNKS_PER_TILE>
void project(InputType *__restrict__ input,
               float* output,
               uint32_t channels, uint32_t features, uint32_t output_dims,
               uint32_t seed, uint32_t num_feature_tiles) {

    if (output_dims % (WARP_SIZE * CHUNKS_PER_TILE) != 0) {
        throw invalid_argument(string("Invalid Number of JL dimensions it has to be a multiple of ") + to_string(WARP_SIZE * CHUNKS_PER_TILE) );
    }

    if (channels == 0 || channels > CHUNK_ROW * NUM_BATCHES) {
        throw invalid_argument("Invalid number of channels (has to be in [1, 8 * NUM_BATCHES])");
    }

    uint32_t num_chunks = (output_dims - 1) / CHUNK_COL + 1;
    uint32_t num_jl_tiles = (num_chunks - 1) / CHUNKS_PER_TILE + 1;

    uint32_t feature_tile_size = (features - 1) / (num_feature_tiles) + 1;

    dim3 blockSize(16, WARP_SIZE / 16, CHUNKS_PER_TILE);
    dim3 gridSize(num_jl_tiles, num_feature_tiles , 1);

    project_kernel<InputType, p_type, NUM_BATCHES, CHUNKS_PER_TILE>
            <<<gridSize, blockSize>>>
            (input, output, channels, features, output_dims, seed, feature_tile_size);
}
