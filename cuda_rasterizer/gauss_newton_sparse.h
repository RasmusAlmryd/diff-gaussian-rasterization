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
        int P, int D, int max_coeffs, int width, int height, // max_coeffs = M
        const float* means3D,
        const int* radii,
        const float* dc,
        const float* shs,
        const bool* clamped,
        const float* opacities,
        const float* scales,
        const float* rotations,
        const float scale_modifier,
        const float* cov3Ds,
        const float* viewmatrix,
        const float* projmatrix,
        const float tan_fovx, const float tan_fovy,
        const float* campos,
        bool antialiasing,
        
        float* x,   // Is named delta in init.py : Check argument position.
        const float* cache, // cached values for gradient calculations (T, ar[3], contributes)
        const uint64_t* cache_indices,
        const uint32_t num_cache_entries, // Number of non 
        float* residuals,
        const bool* tiles_touched,
        const uint32_t N, // number of parameters
        const uint32_t M,  // number of residuals
        const uint32_t num_views, //Number of views that fit in memory
        bool debug
    );
    
}

#endif