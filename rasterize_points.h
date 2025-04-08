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

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

	
std::tuple<int, int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& dc,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const int num_views,
	const int view_index,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dc,
	const torch::Tensor& sh,
	const torch::Tensor& dL_dout_invdepth,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const int K,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int B,
	const torch::Tensor& sampleBuffer,
	const torch::Tensor& residualBuffer,
	const bool antialiasing,
	const int num_views,
	const int view_index,
	const bool debug);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);

void adamUpdate(
	torch::Tensor &param,
	torch::Tensor &param_grad,
	torch::Tensor &exp_avg,
	torch::Tensor &exp_avg_sq,
	torch::Tensor &visible,
	const float lr,
	const float b1,
	const float b2,
	const float eps,
	const uint32_t N,
	const uint32_t M
);

// typedef struct{
// 	int image_height;
// 	int image_width;
// 	float tanfovx ;
// 	float tanfovy ;
// 	const torch::Tensor &bg ;
// 	float scale_modifier ;
// 	const torch::Tensor &viewmatrix ;
// 	const torch::Tensor &projmatrix ;
// 	int sh_degree ;
// 	const torch::Tensor &campos;
// 	bool prefiltered ;
// 	bool debug ;
// 	bool antialiasing ;
// 	int num_views;
// 	int view_index;
// } Raster_settings;

// typedef struct{
// 	int image_height;
// 	int image_width;
// 	float tanfovx ;
// 	float tanfovy ;
// 	const torch::Tensor &bg ;
// 	float scale_modifier ;
// 	const torch::Tensor &viewmatrix ;
// 	const torch::Tensor &projmatrix ;
// 	int sh_degree ;
// 	const torch::Tensor &campos;
// 	bool prefiltered ;
// 	bool debug ;
// 	bool antialiasing ;
// 	int num_views;
// 	int view_index;

// 	Raster_settings(
// 		int image_height,
// 		int image_width,
// 		float tanfovx ,
// 		float tanfovy ,
// 		const torch::Tensor &bg ,
// 		float scale_modifier ,
// 		const torch::Tensor &viewmatrix ,
// 		const torch::Tensor &projmatrix ,
// 		int sh_degree ,
// 		const torch::Tensor &campos,
// 		bool prefiltered ,
// 		bool debug ,
// 		bool antialiasing ,
// 		int num_views,
// 		int view_index,
// 	): image_height(image_height),image_width(image_width),tanfovx(tanfovx),tanfovy(tanfovy),bg(bg),scale_modifier(scale_modifier),viewmatrix(viewmatrix),projmatrix(projmatrix),sh_degree(sh_degree),campos(campos),prefiltered(prefiltered),debug(debug),antialiasing(antialiasing),num_views(num_views),view_index(view_index){}
// } Raster_settings;

// struct Raster_settings{
// 	int image_height;
// 	int image_width;
// 	float tanfovx ;
// 	float tanfovy ;
// 	const torch::Tensor &bg ;
// 	float scale_modifier ;
// 	const torch::Tensor &viewmatrix ;
// 	const torch::Tensor &projmatrix ;
// 	int sh_degree ;
// 	const torch::Tensor &campos;
// 	bool prefiltered ;
// 	bool debug ;
// 	bool antialiasing ;
// 	int num_views;
// 	int view_index;

// 	Raster_settings(
// 		int image_height,
// 		int image_width,
// 		float tanfovx ,
// 		float tanfovy ,
// 		const torch::Tensor &bg ,
// 		float scale_modifier ,
// 		const torch::Tensor &viewmatrix ,
// 		const torch::Tensor &projmatrix ,
// 		int sh_degree ,
// 		const torch::Tensor &campos,
// 		bool prefiltered ,
// 		bool debug ,
// 		bool antialiasing ,
// 		int num_views,
// 		int view_index,
// 	): image_height(image_height),image_width(image_width),tanfovx(tanfovx),tanfovy(tanfovy),bg(bg),scale_modifier(scale_modifier),viewmatrix(viewmatrix),projmatrix(projmatrix),sh_degree(sh_degree),campos(campos),prefiltered(prefiltered),debug(debug),antialiasing(antialiasing),num_views(num_views),view_index(view_index){}
// };

struct Raster_settings {
    int image_height;
    int image_width;
    float tanfovx;
    float tanfovy;
    torch::Tensor bg;
    float scale_modifier;
    torch::Tensor viewmatrix;
    torch::Tensor projmatrix;
    int sh_degree;
    torch::Tensor campos;
    bool prefiltered;
    bool debug;
    bool antialiasing;
    int num_views;
    int view_index;

    Raster_settings(
        int image_height,
        int image_width,
        float tanfovx,
        float tanfovy,
        torch::Tensor bg,
        float scale_modifier,
        torch::Tensor viewmatrix,
        torch::Tensor projmatrix,
        int sh_degree,
        torch::Tensor campos,
        bool prefiltered,
        bool debug,
        bool antialiasing,
        int num_views,
        int view_index
    )
    : image_height(image_height),
      image_width(image_width),
      tanfovx(tanfovx),
      tanfovy(tanfovy),
      bg(std::move(bg)),
      scale_modifier(scale_modifier),
      viewmatrix(std::move(viewmatrix)),
      projmatrix(std::move(projmatrix)),
      sh_degree(sh_degree),
      campos(std::move(campos)),
      prefiltered(prefiltered),
      debug(debug),
      antialiasing(antialiasing),
      num_views(num_views),
      view_index(view_index)
    {}
};


void gaussNewtonUpdate(

	int P, int D, int max_coeffs, int width, int height, // max_coeffs = M
	const torch::Tensor &means3D,
	const torch::Tensor &radii,
	const torch::Tensor &dc,
	const torch::Tensor &shs,
	const torch::Tensor &clamped,
	const torch::Tensor &opacities,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const float scale_modifier,
	const torch::Tensor cov3Ds,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, float tan_fovy,
	const torch::Tensor &campos,
    bool antialiasing,
	const std::vector<Raster_settings> &settings,

    
    torch::Tensor &x,   // Is named delta in init.py : Check argument position.
    torch::Tensor &sparse_J_values,
    torch::Tensor &sparse_J_indices,
    torch::Tensor &sparse_J_p_sum,
	torch::Tensor &loss_residuals,
    float gamma,
    float alpha,
    torch::Tensor &tiles_touched,
    const uint32_t N, // number of parameters
    const uint32_t M,  // number of residuals
    const uint32_t sparse_J_entries
);




void gaussNewtonUpdateSimple(
	torch::Tensor &x,
	const torch::Tensor &J,
	const torch::Tensor &residuals,
	const uint32_t N, // number of parameters
    const uint32_t M  // number of residuals
);

// void sumNumContrib(
//     int W, int H,
//     const int* n_contrib,
//     torch::Tensor sum
// )
	
torch::Tensor
fusedssim(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2
);

torch::Tensor
fusedssim_backward(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &dL_dmap
);
