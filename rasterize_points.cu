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



#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/adam.h"
#include "cuda_rasterizer/gauss_newton.h"
#include "cuda_rasterizer/gauss_newton_simple.h"
#include "rasterize_points.h"
#include <fstream>
#include <string>
#include <functional>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

// struct Raster_settings raster_settings;

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

// 	// Raster_settings(
// 	// 	const int image_heigh,
// 	// 	const int image_width,
// 	// 	const float tanfovx,
// 	// 	const float tanfovy,
// 	// 	const torch::Tensor &bg,
// 	// 	const float scale_modifier,
// 	// 	const torch::Tensor &viewmatrix,
// 	// 	const torch::Tensor &projmatrix,
// 	// 	const int sh_degree,
// 	// 	const torch::Tensor &campo,
// 	// 	const bool prefiltered,
// 	// 	const bool debug,
// 	// 	const bool antialiasing,
// 	// 	const int num_view,
// 	// 	const int view_inde,
// 	// )()image_height(image_height), image_width(image_width), tanfovx()tanfovy()bg()scale_modifier()viewmatrix()projmatrix()sh_degree()campo()prefiltered()debug()antialiasing()num_view()view_inde()
// } ;

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


std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::function<int*(size_t N)> resizeIntFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return t.contiguous().data_ptr<int>();
    };
    return lambda;
}

std::function<float*(size_t N)> resizeFloatFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return t.contiguous().data_ptr<float>();
    };
    return lambda;
}

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
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
  
  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);
  
  torch::Tensor out_color = torch::full({NUM_CHANNELS_3DGS, H, W}, 0.0, float_opts);
  torch::Tensor out_invdepth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor clamped = torch::full({P, 3}, 0, means3D.options().dtype(at::kBool));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  torch::Tensor sampleBuffer = torch::empty({0}, options.device(device));
  torch::Tensor residualBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  std::function<char*(size_t)> sampleFunc = resizeFunctional(sampleBuffer);
  std::function<char*(size_t)> residualFunc = resizeFunctional(residualBuffer);
  
  int rendered = 0;
  int num_residuals = 0;
  int num_buckets = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  auto tup = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
		sampleFunc,
		residualFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		dc.contiguous().data_ptr<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		num_views,
		view_index,
		out_color.contiguous().data<float>(),
		out_invdepth.contiguous().data<float>(),
		antialiasing,
		radii.contiguous().data<int>(),
		clamped.contiguous().data<bool>(),
		debug);
		
		rendered = std::get<0>(tup);
		num_residuals = std::get<1>(tup);
		num_buckets = std::get<2>(tup);
  }
  return std::make_tuple(rendered, num_residuals, num_buckets, out_color, out_invdepth, radii, clamped, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, residualBuffer);
}

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
	const int K, // num sparse residual
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int B,
	const torch::Tensor& sampleBuffer,
	const torch::Tensor& residualBuffer,
	const bool antialiasing,
	const int num_views,
	const int view_index,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS_3DGS}, means3D.options());
  torch::Tensor dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_ddc = torch::zeros({P, 1, 3}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options()); // quats {P, 3, 3}

  int num_images = num_views;

  torch::Tensor dr_dxs = torch::zeros({P, H, W , 13, num_images}, means3D.options()); //accessed in right order?
  torch::Tensor residual_index = torch::zeros({K,num_images}, means3D.options().dtype(torch::kUInt64));
  torch::Tensor p_sum = torch::zeros({P,num_images}, means3D.options().dtype(torch::kUInt32));
  torch::Tensor cov3D = torch::zeros({P, 6}, means3D.options());

  if(P != 0)
  {  
	CudaRasterizer::Rasterizer::backward(P, degree, M, R, B, K,
		background.contiguous().data<float>(),
		W, H, 
		means3D.contiguous().data<float>(),
		dc.contiguous().data<float>(),
		sh.contiguous().data<float>(),
		colors.contiguous().data<float>(),
		opacities.contiguous().data<float>(),	
		scales.data_ptr<float>(),
		scale_modifier,
		rotations.data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		radii.contiguous().data<int>(),
		reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(sampleBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(residualBuffer.contiguous().data_ptr()),
		dL_dout_color.contiguous().data<float>(),
		dL_dout_invdepth.contiguous().data<float>(),
		dL_dmeans2D.contiguous().data<float>(),
		dL_dconic.contiguous().data<float>(),  
		dL_dopacity.contiguous().data<float>(),
		dL_dcolors.contiguous().data<float>(),
		dL_dinvdepths.contiguous().data<float>(),
		dL_dmeans3D.contiguous().data<float>(),
		dL_dcov3D.contiguous().data<float>(),
		dL_ddc.contiguous().data<float>(),
		dL_dsh.contiguous().data<float>(),
		dL_dscales.contiguous().data<float>(),
		dL_drotations.contiguous().data<float>(),
		dr_dxs.contiguous().data<float>(),
		residual_index.contiguous().data<uint64_t>(),
		p_sum.contiguous().data<uint32_t>(),
		cov3D.contiguous().data<float>(),
		antialiasing,
		num_views,
		view_index,
		debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_ddc, dL_dsh, dL_dscales, dL_drotations, dr_dxs, residual_index, p_sum, cov3D);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

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
){
	ADAM::adamUpdate(
		param.contiguous().data<float>(),
		param_grad.contiguous().data<float>(),
		exp_avg.contiguous().data<float>(),
		exp_avg_sq.contiguous().data<float>(),
		visible.contiguous().data<bool>(),
		lr,
		b1,
		b2,
		eps,
		N,
		M);
}

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
	const int num_views,
	// const std::vector<Raster_settings> &settings,

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
){

	// for(Raster_settings r: settings){
	// 	printf("width: %d, height: %d\n", r.image_width, r.image_height);
	// }

	// std::vector<GaussNewton::raster_settings> GN_settings;
    // for(Raster_settings r: settings){
    //     GaussNewton::raster_settings s;
    //     s.image_height = r.image_height;
    //     s.image_width = r.image_width;
    //     s.tanfovx = r.tanfovx;
    //     s.tanfovy = r.tanfovy;
    //     s.bg = r.bg.contiguous().data<float>();
    //     s.scale_modifier = r.scale_modifier;
    //     s.viewmatrix = r.viewmatrix.contiguous().data<float>();
    //     s.projmatrix = r.projmatrix.contiguous().data<float>();
    //     s.sh_degree = r.sh_degree;
    //     s.campos = r.campos.contiguous().data<float>();
    //     s.prefiltered = r.prefiltered;
    //     s.debug = r.debug;
    //     s.antialiasing = r.antialiasing;
    //     s.num_views = r.num_views;
    //     s.view_index = r.view_index;
    //     GN_settings.push_back(s);
    // }

	// return;
	GaussNewton::gaussNewtonUpdate(
		P, D, max_coeffs, width, height, // max_coeffs = M
		means3D.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		dc.contiguous().data<float>(),
		shs.contiguous().data<float>(),
		clamped.contiguous().data<bool>(),
		opacities.contiguous().data<float>(),
		scales.contiguous().data<float>(),
		rotations.contiguous().data<float>(),
		scale_modifier,
		cov3Ds.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		tan_fovx, tan_fovy,
		campos.contiguous().data<float>(),
		antialiasing,
		// GN_settings.data(),
		// GN_settings.size(),
		num_views,

		x.contiguous().data<float>(), 
		sparse_J_values.contiguous().data<float>(),
		sparse_J_indices.contiguous().data<uint64_t>(),
		sparse_J_p_sum.contiguous().data<uint32_t>(),
		loss_residuals.contiguous().data<float>(),
		gamma,
		alpha,
		tiles_touched.contiguous().data<bool>(),
		N, 
		M, 
		sparse_J_entries);
}

void gaussNewtonUpdateSimple(
	torch::Tensor &x,
	const torch::Tensor &J,
	const torch::Tensor &residuals,
	const uint32_t N, // number of parameters
    const uint32_t M  // number of residuals
){
	GaussNewtonSimple::gaussNewtonUpdate(
		x.contiguous().data<float>(), 
		J.contiguous().data<float>(), 
		residuals.contiguous().data<float>(), 
		N, 
		M
	);
}
