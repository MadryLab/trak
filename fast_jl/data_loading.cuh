#include <cuda_fp16.h>

template <typename InputType>
__device__ half convert(InputType t);

template<>
__device__ half convert<half>(half t) {return t;}

template<>
__device__ half convert<float>(float t) {return __float2half(t);}

template<typename InputType, uint32_t CHUNKS_PER_TILE>
__device__
void load_into_shared_memory(
        InputType* __restrict__ input,
        half input_buffer[CHUNKS_PER_TILE][CHUNK_ROW][16],
        uint32_t channels,
        uint32_t features,
        uint32_t feature_tile_size,
        uint32_t iteration,
        uint32_t batch) {


    for (uint32_t k=0; k < 4; k++){
        uint32_t current_col = (
                threadIdx.x
                + iteration
                + threadIdx.z * (blockDim.x             ));

        uint32_t tile_offset = + blockIdx.y  * (feature_tile_size);

        uint32_t current_row = (
                threadIdx.y +
                blockDim.y * k
                + 8 * batch);

        InputType *my_input = (
                input
                + current_row * features      // row
                + current_col + tile_offset // column
        );

        half value;

        if (
                current_col + tile_offset >= features // Check if we go out of bounds of the matrix (Column wise)
                || current_col >= feature_tile_size   // Check if we go outside of the input tile this thread block is responsible for
                ||current_row >= channels             // Check if we go out of bounds of the matrix (Row wise)
                ) {
            value = __float2half(0.0f);
        } else {
            value = convert<InputType>(*my_input);
        }
        input_buffer[threadIdx.z][k * blockDim.y + threadIdx.y][threadIdx.x] = value;
    }
}
