//
// Created by guillaumeleclerc on 1/11/23.
//

#ifndef MEGAFASTJL_JL_RANDOM_H
#define MEGAFASTJL_JL_RANDOM_H

#include <curand_kernel.h>

#include "types.cuh"

namespace jl_random {
    __device__ uint32_t init(curandStateXORWOW_t &random_state, uint32_t column, uint32_t feature_tile, uint32_t output_dim, uint32_t seed) {

        uint32_t my_seed = seed
                           + 658105236 * column
                           + 359124577 * feature_tile;

        curand_init(my_seed,
                0, 0, &random_state);
        return my_seed;
    }

    template<ProjectionType p_type>
    __device__ void generate_factors_fragment(half* dst, curandStateXORWOW_t  &random_state);

    template<>
    __device__ void generate_factors_fragment<Rademacher>(half* dst, curandStateXORWOW_t  &random_state) {
        half2 *__restrict__ dst2 = reinterpret_cast<half2*>(dst);
        auto random_bits = curand(&random_state);
        for (uint32_t x = 0; x < 8; x++) {
            half2 toto;
            if (random_bits & 1) {
                toto.x =__float2half(1.0f);
            } else {
                toto.x =__float2half(-1.0f);
            }
            random_bits >>= 1;
            if (random_bits & 1) {
                toto.y =__float2half(1.0f);
            } else {
                toto.y =__float2half(-1.0f);
            }
            dst2[threadIdx.x + threadIdx.y * 16] = toto;
            random_bits >>= 1;
            dst2 += 32;
        }
    }

    template<>
    __device__ void generate_factors_fragment<Normal>(half* dst, curandStateXORWOW_t  &random_state) {
        half2 *__restrict__ dst2 = reinterpret_cast<half2*>(dst);
        half2 toto;
        for (uint32_t x = 0; x < 8; x++) {
            toto.x = __float2half(curand_normal(&random_state));
            toto.y = __float2half(curand_normal(&random_state));
            dst2[threadIdx.x + threadIdx.y * 16] = toto;
            dst2 += 32;
        }
    }


}



#endif //MEGAFASTJL_JL_RANDOM_H
