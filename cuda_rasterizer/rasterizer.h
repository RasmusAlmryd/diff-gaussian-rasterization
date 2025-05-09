/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include <glm/glm.hpp>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static std::tuple<int, int, int> forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			std::function<char* (size_t)> sampleBuffer,
			std::function<char* (size_t)> residualBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* dc,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			const int num_views,
			const int view_index,
			float* out_color,
			float* depth,
			bool antialiasing,
			int* radii = nullptr,
			bool* clamped = nullptr,
			bool GN_enabled = false,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R, int B, int K,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* dc,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			char* sample_buffer,
			char* residual_buffer,
			const float* dL_dpix,
			const float* dL_invdepths,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dinvdepth,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_ddc,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dr_dxs,
			uint64_t* residual_index,
			uint32_t* p_sum,
			float* cov3D,
			float* conic_o,
			bool antialiasing,
			const int num_views,
			const int view_index,
			bool GN_enabled,
			bool debug);
	};
};

#endif