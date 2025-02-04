#ifndef CUDA_RASTERIZER_GN_H_INCLUDED
#define CUDA_RASTERIZER_GN_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/extension.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace GaussNewton {

void GaussNewtonUpdate(
    float* x,
    const float* J,
    float gamma,
    float alpha,
    const bool* tiles_touched,
    const uint32_t N,
    const uint32_t M);
}

#endif