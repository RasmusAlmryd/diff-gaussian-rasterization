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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <stdexcept>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

__device__ inline float evaluate_opacity_factor(const float dx, const float dy, const float4 co) {
	return 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
}

template<uint32_t PATCH_WIDTH, uint32_t PATCH_HEIGHT>
__device__ inline float max_contrib_power_rect_gaussian_float(
	const float4 co, 
	const float2 mean, 
	const glm::vec2 rect_min,
	const glm::vec2 rect_max,
	glm::vec2& max_pos)
{
	const float x_min_diff = rect_min.x - mean.x;
	const float x_left = x_min_diff > 0.0f;
	// const float x_left = mean.x < rect_min.x;
	const float not_in_x_range = x_left + (mean.x > rect_max.x);

	const float y_min_diff = rect_min.y - mean.y;
	const float y_above =  y_min_diff > 0.0f;
	// const float y_above = mean.y < rect_min.y;
	const float not_in_y_range = y_above + (mean.y > rect_max.y);

	max_pos = {mean.x, mean.y};
	float max_contrib_power = 0.0f;

	if ((not_in_y_range + not_in_x_range) > 0.0f)
	{
		const float px = x_left * rect_min.x + (1.0f - x_left) * rect_max.x;
		const float py = y_above * rect_min.y + (1.0f - y_above) * rect_max.y;

		const float dx = copysign(float(PATCH_WIDTH), x_min_diff);
		const float dy = copysign(float(PATCH_HEIGHT), y_min_diff);

		const float diffx = mean.x - px;
		const float diffy = mean.y - py;

		const float rcp_dxdxcox = __frcp_rn(PATCH_WIDTH * PATCH_WIDTH * co.x); // = 1.0 / (dx*dx*co.x)
		const float rcp_dydycoz = __frcp_rn(PATCH_HEIGHT * PATCH_HEIGHT * co.z); // = 1.0 / (dy*dy*co.z)

		const float tx = not_in_y_range * __saturatef((dx * co.x * diffx + dx * co.y * diffy) * rcp_dxdxcox);
		const float ty = not_in_x_range * __saturatef((dy * co.y * diffx + dy * co.z * diffy) * rcp_dydycoz);
		max_pos = {px + tx * dx, py + ty * dy};
		
		const float2 max_pos_diff = {mean.x - max_pos.x, mean.y - max_pos.y};
		max_contrib_power = evaluate_opacity_factor(max_pos_diff.x, max_pos_diff.y, co);
	}

	return max_contrib_power;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float4* __restrict__ conic_opacity,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid,
	int2* rects)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		const uint32_t offset_to = offsets[idx];
		uint2 rect_min, rect_max;

		if(rects == nullptr)
			getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
		else
			getRect(points_xy[idx], rects[idx], rect_min, rect_max, grid);

		const float2 xy = points_xy[idx];
		const float4 co = conic_opacity[idx];
		const float opacity_threshold = 1.0f / 255.0f;
		const float opacity_factor_threshold = logf(co.w / opacity_threshold);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				const glm::vec2 tile_min(x * BLOCK_X, y * BLOCK_Y);
				const glm::vec2 tile_max((x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1);

				glm::vec2 max_pos;
				float max_opac_factor = 0.0f;
				max_opac_factor = max_contrib_power_rect_gaussian_float<BLOCK_X-1, BLOCK_Y-1>(co, xy, tile_min, tile_max, max_pos);
				
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				// printf("[%llu] max_opac: %f, opac threshold: %f, conic_w: %f\n",key, max_opac_factor, opacity_factor_threshold, co.w);
				if (max_opac_factor <= opacity_factor_threshold) {
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}

		for (; off < offset_to; ++off) {
			uint64_t key = (uint32_t) -1;
			key <<= 32;
			const float depth = FLT_MAX;
			key |= *((uint32_t*)&depth);
			gaussian_values_unsorted[off] = static_cast<uint32_t>(-1);
			gaussian_keys_unsorted[off] = key;
		}
	}
	// else{
	// 	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
	// 	const uint32_t offset_to = offsets[idx];
	// 	for (; off < offset_to; ++off) {
	// 		uint64_t key = (uint32_t) -1;
	// 		key <<= 32;
	// 		const float depth = FLT_MAX;
	// 		key |= *((uint32_t*)&depth);
	// 		gaussian_values_unsorted[off] = static_cast<uint32_t>(-1);
	// 		gaussian_keys_unsorted[off] = key;
	// 	}
	// }
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// printf("idx: %d, %d \n", idx, point_list_keys[idx]);
	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	bool valid_tile = currtile != (uint32_t) -1;
	// printf("idx: %d, key: %llu, current tile: %zu, valid: %d\n",idx, key, currtile, valid_tile);
	// if(!valid_tile){
	// 	printf("not valid tile");
	// }

	if (idx == 0 ){
		if(valid_tile){
			ranges[currtile].x = 0;
		}
	}
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			if (valid_tile) 
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1 && valid_tile)
		ranges[currtile].y = L;
	
}

// for each tile, see how many buckets/warps are needed to store the state
__global__ void perTileBucketCount(int T, uint2* ranges, uint32_t* bucketCount) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= T)
		return;
	
	uint2 range = ranges[idx];
	int num_splats = range.y - range.x;
	int num_buckets = (num_splats + 31) / 32;
	bucketCount[idx] = (uint32_t) num_buckets;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.actual_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	int* dummy = nullptr;
	int* wummy = nullptr;
	cub::DeviceScan::InclusiveSum(nullptr, img.scan_size, dummy, wummy, N);
	obtain(chunk, img.contrib_scan, img.scan_size, 128);

	obtain(chunk, img.max_contrib, N, 128);
	obtain(chunk, img.pixel_colors, N * NUM_CHANNELS_3DGS, 128);
	obtain(chunk, img.pixel_invDepths, N, 128);
	obtain(chunk, img.bucket_count, N, 128);
	obtain(chunk, img.bucket_offsets, N, 128);
	cub::DeviceScan::InclusiveSum(nullptr, img.bucket_count_scan_size, img.bucket_count, img.bucket_count, N);
	obtain(chunk, img.bucket_count_scanning_space, img.bucket_count_scan_size, 128);

	return img;
}

CudaRasterizer::SampleState CudaRasterizer::SampleState::fromChunk(char *& chunk, size_t C) {
	SampleState sample;
	obtain(chunk, sample.bucket_to_tile, C * BLOCK_SIZE, 128);
	obtain(chunk, sample.T, C * BLOCK_SIZE, 128);
	obtain(chunk, sample.ar, NUM_CHANNELS_3DGS * C * BLOCK_SIZE, 128);
	obtain(chunk, sample.ard, C * BLOCK_SIZE, 128);
	return sample;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);

	obtain(chunk, binning.gaussian_contrib, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, binning.gaussian_contrib_scan_size, binning.gaussian_contrib, binning.gaussian_contrib, P);
	obtain(chunk, binning.gaussian_contrib_scan_space, binning.gaussian_contrib_scan_size, 128);

	return binning;
}

CudaRasterizer::ResidualState CudaRasterizer::ResidualState::fromChunk(char*& chunk, size_t R)
{
	ResidualState residual;
	obtain(chunk, residual.list_keys_temp, R, 128);
	obtain(chunk, residual.list_values_temp, R, 128);
	uint64_t* dummy = nullptr;
	float* wummy = nullptr;
	cub::DeviceRadixSort::SortPairs(
		nullptr, residual.sorting_size, 
		residual.list_keys_temp, dummy,
		residual.list_values_temp, wummy, R);

	obtain(chunk, residual.sorting_space, residual.sorting_size, 128);

	
	return residual;
}

__global__ void zero(int N, int* space)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx >= N)
		return;
	space[idx] = 0;
}

__global__ void set(int N, uint32_t* where, int* space)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx >= N)
		return;

	int off = (idx == 0) ? 0 : where[idx-1];

	space[off] = 1;
}

__global__ void allEqual(int N, int num, int* space, bool* result){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx >= N)
		return;

	if(space[idx] != num){
		*result = false;
	}
}


// calculate number of pixels/residuals a duplicate gaussian contributes to
__global__
void perGaussianContribCount(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ n_contrib,
	const uint32_t* __restrict__ max_contrib,
	int W, int H,
	uint32_t* gaussian_contrib
){


	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	// bool done = !inside;
	if(!inside) return;

	// Load start/end range of IDs to process in bit sorted list.
	uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
	uint2 range = ranges[tile_id];
	int toDo = range.y - range.x;

	float last_contributor = n_contrib[pix_id];

	// loop over all gaussians contributing to this pixel
	for(int i = 0; i < toDo; i++){
		if (i >= last_contributor) return;
		int splat_idx_global = range.x + i;
		// if(splat_idx_global == 0){
		// 	  printf("gaussina contrib: %d , pix id: %d, tile id: %d, last contrib: %f, last contrib d: %d, max contrib: %d\n", i, pix_id, tile_id, last_contributor,  n_contrib[pix_id], max_contrib[tile_id]);
		// }
		atomicAdd(&gaussian_contrib[splat_idx_global], 1);
	}
	
}


__global__
void sortByGaussians(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ n_contrib,
	int W, int H,
	uint32_t* gaussian_contrib,
	float* dr_dG,
	int* residual_index,
	int* p_sum)
{

}


// Forward rendering procedure for differentiable rasterization
// of Gaussians.
std::tuple<int,int,int> CudaRasterizer::Rasterizer::forward(
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
	float* invdepth,
	bool antialiasing,
	int* radii,
	bool* clamped,
	bool GN_enabled,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	// CHECK_CUDA(cudaMemset(chunkptr, 0, chunk_size), P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS_3DGS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// for(int i = 0; i < P; i++){
	// 	uint32_t temp;
	// 	CHECK_CUDA(cudaMemcpy(&temp, &geomState.tiles_touched[i], sizeof(uint32_t), cudaMemcpyDeviceToHost), debug);
	// 	printf("tiles touched[%d]: %d\n", i, temp);
	// }

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		dc,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		antialiasing
	), debug)

	// bool any_visible;
	// bool* all_zero;
	// cudaMalloc(&all_zero, sizeof(bool));
	// cudaMemset(all_zero, 1, sizeof(bool));
	// allEqual<<<(P + 255) / 256, 256>>>(P, 0, radii, all_zero);
	// cudaMemcpy(&any_visible, all_zero, sizeof(bool), cudaMemcpyDeviceToHost);
	// any_visible = !any_visible;
	// printf("any_visible: %d\n", any_visible);
	// if(!any_visible){
	// 	// return std::make_tuple(num_rendered, num_residuals, bucket_sum);
	// 	return std::make_tuple(0, 0, 0);
	// }

	// cudaFree(all_zero);

	// store clamped values for use in gauss_newton optimizer
	CHECK_CUDA(cudaMemcpy(clamped, geomState.clamped, P * 3 * sizeof(bool), cudaMemcpyDeviceToDevice), debug);

	// for(int i = 0; i < P; i++){
	// 	int temp;
	// 	CHECK_CUDA(cudaMemcpy(&temp, &radii[i], sizeof(int), cudaMemcpyDeviceToHost), debug);
	// 	printf("radii[%d]: %d\n", i, temp);
	// }

	// for(int i = 0; i < P; i++){
	// 	uint32_t temp;
	// 	CHECK_CUDA(cudaMemcpy(&temp, &geomState.tiles_touched[i], sizeof(uint32_t), cudaMemcpyDeviceToHost), debug);
	// 	printf("tiles touched[%d]: %d\n", i, temp);
	// }

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// for(int i = 0; i < P; i++){
	// 	uint32_t temp;
	// 	CHECK_CUDA(cudaMemcpy(&temp, &geomState.point_offsets[i], sizeof(uint32_t), cudaMemcpyDeviceToHost), debug);
	// 	printf("prefix sum[%d]: %d\n", i, temp);
	// }

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	// printf("forward: num_rendered: %d\n", num_rendered);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.conic_opacity,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid,
		nullptr)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// for(int i = 0; i < num_rendered; i++){
	// 	uint64_t temp;
	// 	CHECK_CUDA(cudaMemcpy(&temp, &binningState.point_list_keys[i], sizeof(uint64_t), cudaMemcpyDeviceToHost), debug);
	// 	uint32_t tile = temp >> 32;
	// 	printf("key[%d]: full: %llu, tile: %zu \n", i, temp, tile);
	// }

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// for(int i = 0; i < tile_grid.x * tile_grid.y; i++){
	// 	uint2 temp;
	// 	CHECK_CUDA(cudaMemcpy(&temp, &imgState.ranges[i], sizeof(uint2), cudaMemcpyDeviceToHost), debug);
	// 	printf("range[%d]: x: %d, y: %d\n", i, temp.x, temp.y);
	// }

 	// bucket count
	int num_tiles = tile_grid.x * tile_grid.y;
	perTileBucketCount<<<(num_tiles + 255) / 256, 256>>>(num_tiles, imgState.ranges, imgState.bucket_count);
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(imgState.bucket_count_scanning_space, imgState.bucket_count_scan_size, imgState.bucket_count, imgState.bucket_offsets, num_tiles), debug)
	unsigned int bucket_sum;
	CHECK_CUDA(cudaMemcpy(&bucket_sum, imgState.bucket_offsets + num_tiles - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost), debug);
	// create a state to store. size is number is the total number of buckets * block_size
	size_t sample_chunk_size = required<SampleState>(bucket_sum);
	char* sample_chunkptr = sampleBuffer(sample_chunk_size);
	SampleState sampleState = SampleState::fromChunk(sample_chunkptr, bucket_sum);


	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(cudaMemset(binningState.gaussian_contrib, 0, num_rendered * sizeof(uint32_t)), debug);
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		imgState.bucket_offsets, sampleState.bucket_to_tile,
		sampleState.T, sampleState.ar, sampleState.ard,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.max_contrib,
		imgState.actual_contrib,
		binningState.gaussian_contrib,
		background,
		out_color,
		geomState.depths,
		invdepth,
		GN_enabled), debug)

	// printf("render");

	CHECK_CUDA(cudaMemcpy(imgState.pixel_colors, out_color, sizeof(float) * width * height * NUM_CHANNELS_3DGS, cudaMemcpyDeviceToDevice), debug);
	CHECK_CUDA(cudaMemcpy(imgState.pixel_invDepths, invdepth, sizeof(float) * width * height, cudaMemcpyDeviceToDevice), debug);

	//gaussian_contrib len = num_rendered (duplicate gaussians)
	int num_residuals = 0;

	if(GN_enabled){
		if(num_rendered > 0){
			// CHECK_CUDA(cudaMemset(binningState.gaussian_contrib, 0, num_rendered * sizeof(uint32_t)), debug);
	
			// // printf("num_rendered: %d", num_rendered);
	
			// perGaussianContribCount<<<tile_grid, block >>>(
			// 	imgState.ranges,
			// 	imgState.n_contrib,
			// 	width, height,
			// 	binningState.gaussian_contrib
			// );
	
			// printf("per gaussian contrib");
	
			
			// uint2 test_range;
			// CHECK_CUDA(cudaMemcpy(&test_range, &imgState.ranges[0], sizeof(uint2), cudaMemcpyDeviceToHost), debug);
			
			// printf("range of tile #1: x: %d y: %d, range: %d \n", test_range.x, test_range.y, test_range.y-test_range.x);
			
			// int pix_contrib;
			// CHECK_CUDA(cudaMemcpy(&pix_contrib, &imgState.n_contrib[0], sizeof(int), cudaMemcpyDeviceToHost), debug);
			// printf("n_contrib[0]: %d \n", pix_contrib);
			
			// CHECK_CUDA(cudaMemcpy(&pix_contrib, &imgState.n_contrib[70], sizeof(int), cudaMemcpyDeviceToHost), debug);
			// printf("n_contrib[70]: %d \n", pix_contrib);
			
			// to prefix sum over gaussian_contrib
			// CHECK_CUDA(cub::DeviceScan::InclusiveSum(binningState.gaussian_contrib_scan_space, binningState.gaussian_contrib_scan_size, binningState.gaussian_contrib, binningState.gaussian_contrib, num_rendered), debug)
			
			
			// CHECK_CUDA(cudaMemcpy(&num_residuals, binningState.gaussian_contrib + num_rendered - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
			// printf("num_residuals: %d \n", num_residuals);
			// printf("tile_grid: x: %d y: %d \n", tile_grid.x, tile_grid.y);
			
			// printf("inclusive sum");
			
			
			// CHECK_CUDA(cudaMemset(binningState.gaussian_contrib, 0, num_rendered * sizeof(uint32_t)), debug);
			
			// perGaussianContribCount<<<tile_grid, block >>>(
			// 	imgState.ranges,
			// 	imgState.n_contrib,
			// 	imgState.max_contrib,
			// 	width, height,
			// 	binningState.gaussian_contrib
			// );
			
			// int gauss_residuals;
			// for (int i = 0; i < 10; i+=1){
			// 	CHECK_CUDA(cudaMemcpy(&gauss_residuals, &binningState.gaussian_contrib[i], sizeof(int), cudaMemcpyDeviceToHost), debug);
			
			// 	printf("number of residuals contributed to by gaussian #%d: %d\n",i, gauss_residuals);
			// }

			
			// CHECK_CUDA(cub::DeviceScan::InclusiveSum(binningState.gaussian_contrib_scan_space, binningState.gaussian_contrib_scan_size, binningState.gaussian_contrib, binningState.gaussian_contrib, num_rendered), debug)
			
		
			// CHECK_CUDA(cudaMemcpy(&num_residuals, binningState.gaussian_contrib + num_rendered - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

			// int contrib;
			// for(int i = 0; i < num_rendered; i++){
			// 	CHECK_CUDA(cudaMemcpy(&contrib, binningState.gaussian_contrib + i, sizeof(int), cudaMemcpyDeviceToHost), debug);
			// 	printf("g: %d, contrib: %d\n", i, contrib);
			// }
			// printf("num_residuals: %d \n", num_residuals);

			// int contrib;
			// int g_id;
			// for(int i = 0; i < num_rendered; i++){
			// 	CHECK_CUDA(cudaMemcpy(&contrib, binningState.gaussian_contrib + i, sizeof(int), cudaMemcpyDeviceToHost), debug);
			// 	CHECK_CUDA(cudaMemcpy(&g_id, binningState.point_list + i, sizeof(int), cudaMemcpyDeviceToHost), debug);
			// 	printf("g: %d [%d], contrib: %d\n", i, g_id, contrib);
			// }

			CHECK_CUDA(cub::DeviceScan::InclusiveSum(binningState.gaussian_contrib_scan_space, binningState.gaussian_contrib_scan_size, binningState.gaussian_contrib, binningState.gaussian_contrib, num_rendered), debug)
			CHECK_CUDA(cudaMemcpy(&num_residuals, binningState.gaussian_contrib + num_rendered - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
			printf("new: num_residuals: %d \n", num_residuals);

			// CHECK_CUDA(cudaMemset(binningState.gaussian_contrib, 0, num_rendered * sizeof(uint32_t)), debug);
			
			// perGaussianContribCount<<<tile_grid, block >>>(
			// 	imgState.ranges,
			// 	imgState.n_contrib,
			// 	imgState.max_contrib,
			// 	width, height,
			// 	binningState.gaussian_contrib
			// );

			// for(int i = 0; i < num_rendered; i++){
			// 	CHECK_CUDA(cudaMemcpy(&contrib, binningState.gaussian_contrib + i, sizeof(int), cudaMemcpyDeviceToHost), debug);
			// 	CHECK_CUDA(cudaMemcpy(&g_id, binningState.point_list + i, sizeof(int), cudaMemcpyDeviceToHost), debug);
			// 	printf("g: %d [%d], contrib: %d\n", i, g_id, contrib);
			// }

			// CHECK_CUDA(cub::DeviceScan::InclusiveSum(binningState.gaussian_contrib_scan_space, binningState.gaussian_contrib_scan_size, binningState.gaussian_contrib, binningState.gaussian_contrib, num_rendered), debug)
			
		
			// CHECK_CUDA(cudaMemcpy(&num_residuals, binningState.gaussian_contrib + num_rendered - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
			// printf("old: num_residuals: %d \n", num_residuals);

			
			// throw std::invalid_argument("temp exception.. REMOVE");
			// printf("num residuals: %d", num_residuals);
			// size_t residual_chunk_size = required<ResidualState>(num_residuals);
			// char* residual_chunkptr = residualBuffer(residual_chunk_size);
			// ResidualState residualStat = ResidualState::fromChunk(residual_chunkptr, num_residuals);
		}
	
	}
	
	// printf("forward done");
	
	return std::make_tuple(num_rendered, num_residuals, bucket_sum);
}




// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R, int B, int K, //P: num gaussian, R: num duplicate gaussians, K: num residuals
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
	char* img_buffer,
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
	bool debug)
{
	// printf("num duplicate gaussians: %d\n", R);
	// printf("num gaussians: %d\n", P);
	// printf("num residuals: %d \n", K);
	// printf("num views: %d\n", num_views);
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);
	SampleState sampleState = SampleState::fromChunk(sample_buffer, B);
	// ResidualState residualState = ResidualState::fromChunk(residual_buffer, K);


	// printf("num residuals: %d\n", K);   //Useful prints for debugging
	// printf("num residuals/num gaussians: %f\n", (float)K / (float)R);



	// int gauss_residuals;
	// CHECK_CUDA(cudaMemcpy(&gauss_residuals, &binningState.gaussian_contrib[6322], sizeof(int), cudaMemcpyDeviceToHost), debug);

	// printf("number of residuals contributed to by gaussian #1: %d\n", gauss_residuals);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height, P, R, B, K, 
		imgState.bucket_offsets,
		sampleState.bucket_to_tile,
		sampleState.T,
		sampleState.ar,
		sampleState.ard,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.max_contrib,
		binningState.gaussian_contrib,
		imgState.pixel_colors,
		imgState.pixel_invDepths,
		dL_dpix,
		dL_invdepths,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepth,
		dr_dxs, 
		residual_index,
		num_views,
		view_index,
		GN_enabled), debug)

	// float test;
	// int test_index;
	// for (int i = 0; i < min(K, 200); i+=1){
	//  	CHECK_CUDA(cudaMemcpy(&test, &dr_dxs[i*5], sizeof(float), cudaMemcpyDeviceToHost), debug);
	// 	CHECK_CUDA(cudaMemcpy(&test_index, &residual_index[i], sizeof(int), cudaMemcpyDeviceToHost), debug);

	// 	int gaussain = (test_index % (P*width*height)) % P;
	// 	int residual = (test_index % (P*width*height)) / P;
	// 	uint2 pos = {(uint32_t)residual % width, (uint32_t)residual / width};
	// 	int view = test_index / (P*width*height);  // View index
	//  	printf("Sparse dr_dxs content #%d [gaussian: %d, residual: (x: %d, y: %d) , view: %d] ar[0]: %g\n",i, gaussain, pos.x, pos.y, view, test);
	// }



	// CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
	// 	binningState.list_sorting_space,
	// 	binningState.sorting_size,
	// 	binningState.point_list_keys_unsorted, binningState.point_list_keys,
	// 	binningState.point_list_unsorted, binningState.point_list,
	// 	num_rendered, 0, 32 + bit), debug);

	// int bit = getHigherMsb(P*width*height);
	//printf("----------------------------------------------------------------\n");
	//printf("SORTED: \n");


	//CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
	//	residualState.sorting_space,
	//	residualState.sorting_size,
	//	residual_index, residualState.list_keys_temp,
	//	dr_dxs, residualState.list_values_temp,
	//	K), debug);
	
	//printf("done sorting");



	// for (int i = 0; i < 400; i+=1){
	// 	CHECK_CUDA(cudaMemcpy(&test, &residualState.list_values_temp[i], sizeof(float), cudaMemcpyDeviceToHost), debug);
	// 	CHECK_CUDA(cudaMemcpy(&test_index, &residualState.list_keys_temp[i], sizeof(int), cudaMemcpyDeviceToHost), debug);

	// 	printf("Sparse dr_dxs content #%d: in index [%d] (pix_id: %d, gauss_id: %d)   : %g,   \n",i, test_index, (test_index / P), (test_index % P), test);
	// }


	// throw std::invalid_argument("temp exception.. REMOVE");

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	const float4* conic_o_ptr = geomState.conic_opacity;

	// store clamped values for use in gauss_newton optimizer
	if(GN_enabled){
		CHECK_CUDA(cudaMemcpy(cov3D, cov3D_ptr, P * 6 * sizeof(float), cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(conic_o, conic_o_ptr, P * sizeof(float4), cudaMemcpyDeviceToDevice), debug);
		return;
	}              



	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		dc,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_dinvdepth,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_ddc,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		antialiasing), debug)
}
