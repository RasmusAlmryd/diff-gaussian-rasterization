#ifndef CUDA_RASTERIZER_GN_H_INCLUDED
#define CUDA_RASTERIZER_GN_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#define CUDA_LAUNCH_BLOCKING 1
#include <glm/glm.hpp>

namespace GaussNewton {

    void gaussNewtonUpdate(
        float* x,   // Is named delta in init.py : Check argument position.
        float* sparse_J_values,
        uint64_t* sparse_J_indices,
        uint32_t* sparse_J_p_sum,
        float* loss_residuals,
        float gamma,
        float alpha,
        const bool* tiles_touched,
        const uint32_t N, // number of parameters
        const uint32_t M,  // number of residuals
        const uint32_t sparse_J_entries
        );

    
}

#endif