#ifndef CUDA_RASTERIZER_GN_H_INCLUDED
#define CUDA_RASTERIZER_GN_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#define CUDA_LAUNCH_BLOCKING 1
#include <glm/glm.hpp>

namespace GaussNewton {

    struct raster_settings{
        int image_height;
        int image_width;
        float tanfovx ;
        float tanfovy ;
        float* bg ;
        float scale_modifier ;
        float* viewmatrix ;
        float* projmatrix ;
        int sh_degree ;
        float* campos;
        bool prefiltered ;
        bool debug ;
        bool antialiasing ;
        int num_views;
        int view_index;
    };

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
	    const float tan_fovx, float tan_fovy,
        const float* campos,
        bool antialiasing,
        const int num_views,
        // const raster_settings* settings,
	    // const size_t num_views,



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