#include "auxiliary.h"
#include "gauss_newton_sparse.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#include <string>
#include <iostream>



#define THREADS_PER_BLOCK 256
#define VALUES_PER_CACHE_ENTIRE 4
#define NUM_PARAMS_PER_GAUSSIAN 59

// __global__ 
// void isnan_test(float *data, bool *result, uint32_t N){
//     __shared__ bool temp[THREADS_PER_BLOCK];
//     uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

//     if(idx >= N) return;

//     temp[threadIdx.x] = isnan(data[idx]);

//     __syncthreads();

//     if (threadIdx.x == 0)
//     {
//         bool block_nan = false;
//         for (int i = 0; i < THREADS_PER_BLOCK; i++)
//         {
//             if(temp[i]){
//                 block_nan = true;
//                 break;
//             }
//         }
//         atomicAdd((int*)result, (int)block_nan);
//     }

//     // bool is_nan = false;
//     // if(idx < size){
//     //     if(isnan(data[idx])){
//     //         is_nan = true;
//     //     }
//     // }

//     // bool block_nan = __syncthreads_or(is_nan);
//     // if(threadIdx.x == 0){
//     //     if(block_nan)
//     //         *result = true;
//     // }
// }

__global__ 
void isnan_test_f(const float *data, bool *result, uint32_t N) {
    // Shared flag for early exit
    __shared__ bool found_nan;

    // Initialize shared flag (only one thread per block)
    if (threadIdx.x == 0) {
        found_nan = false;
    }
    __syncthreads();

    // Get global thread index
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop over data if threads < N
    for (uint32_t i = idx; i < N; i += gridDim.x * blockDim.x) {
        if (isnan(data[i])) {
            found_nan = true;
            break; // Early exit for this thread
        }
    }

    __syncthreads();

    // Set global result (by one thread only)
    if (threadIdx.x == 0 && found_nan) {
        *result = true;
    }
}

__global__ 
void isnan_test_d(const double *data, bool *result, uint32_t N) {
    // Shared flag for early exit
    __shared__ bool found_nan;

    // Initialize shared flag (only one thread per block)
    if (threadIdx.x == 0) {
        found_nan = false;
    }
    __syncthreads();

    // Get global thread index
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop over data if threads < N
    for (uint32_t i = idx; i < N; i += gridDim.x * blockDim.x) {
        if (isnan(data[i])) {
            found_nan = true;
            break; // Early exit for this thread
        }
    }

    __syncthreads();

    // Set global result (by one thread only)
    if (threadIdx.x == 0 && found_nan) {
        *result = true;
    }
}

__global__ 
void dot(float *a, float *b, float *c, uint32_t N){
    __shared__ float temp[THREADS_PER_BLOCK];
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= N) return;

    temp[threadIdx.x] = a[idx] * b[idx];

    __syncthreads();

    if (threadIdx.x == 0)
    {
        float sum = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++)
        {
            sum += temp[i];
        }
        atomicAdd(c, sum);
    }
}

__global__ 
void dot_d(double *a, double *b, double *c, uint32_t N){
    __shared__ double temp[THREADS_PER_BLOCK];
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= N) return;

    temp[threadIdx.x] = a[idx] * b[idx];
    // if(isnan(temp[threadIdx.x])){
    //     printf("IS NAN: idx: %d, a: %g, b: %g\n", idx, a[idx] * b[idx]);

    // }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        double sum = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++)
        {
            sum += (double)temp[i];
        }
        // printf("partial sum: %g", sum);
        atomicAdd(c, sum);
    }
}

__global__
void scalar_divide(const float* numerator, const float* denominator, float* quotient){
    
    if (*denominator != 0.0f) {
        *quotient = *numerator / *denominator;
    } else {
        *quotient = 0.0f;  
    }
}

__global__
void scalar_divide_d(const double* numerator, const double* denominator, double* quotient){
    
    if (*denominator != 0.0f) {
        *quotient = *numerator / *denominator;
    } else {
        *quotient = 0.0f;  
    }
}

__global__
void next_x(float* x, float* p, float* alpha, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    x[idx] = x[idx] + *alpha * p[idx];
}

__global__
void next_x_d(float* x, double* p, double* alpha, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    x[idx] = x[idx] + *alpha * p[idx];
}


__global__
void next_p(float* z, float* p, float* beta, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    float new_val = z[idx] + *beta * p[idx];
    p[idx] = new_val;
}

__global__
void next_p_d(double* z, double* p, double* beta, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    double new_val = z[idx] + *beta * p[idx];
    p[idx] = new_val;
}

__global__
void next_r(float* r, float* Ap, float* alpha, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    float new_val = r[idx] - *alpha * Ap[idx];
    r[idx] = new_val;
} 

__global__
void next_r_d(double* r, double* Ap, double* alpha, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    double new_val = r[idx] - *alpha * Ap[idx];
    r[idx] = new_val;
} 

__global__ void next_z(float* M_precon, float* r, float* z, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    float denominator = M_precon[idx];
    float inv = 1.0 / (denominator + 1e-8f);
    float z_val = inv * r[idx];
    z[idx] = z_val;
}

__global__ void next_z_d(double* M_precon, double* r, double* z, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    double denominator = M_precon[idx];
    double inv = 1.0 / (denominator + 1e-8f);
    double z_val = inv * r[idx];
    z[idx] = z_val;
}

__global__ 
void subtract(float* a, float* b, float* c, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    c[idx] = a[idx] - b[idx];
}

__global__ 
void subtract_d(double* a, double* b, double* c, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    c[idx] = a[idx] - b[idx];
}

__device__ 
float sigmoid_grad(float x){
    float s = sigmoid(x);
    return s * (1-s);
}


__device__ void computeColorFromSH_device(
    int idx, 
    int deg, 
    int max_coeffs, 
    const glm::vec3* means,
    glm::vec3 campos, 
    const float* dc, 
    const float* shs,
    const bool* clamped,
    const glm::vec3* dL_dcolor,
    glm::vec3* dL_dmeans,
    glm::vec3* dL_ddc,
    glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	// glm::vec3* direct_color = ((glm::vec3*)dc);  //we removed + idx here
	// glm::vec3* sh = ((glm::vec3*)shs); //we removed + idx here and * max_coeffs

    glm::vec3* direct_color = ((glm::vec3*)dc) + idx;
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[0];
	// dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	// dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	// dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;
    dL_dRGB.x *= clamped[0] ? 0 : 1;
	dL_dRGB.y *= clamped[1] ? 0 : 1;
	dL_dRGB.z *= clamped[2] ? 0 : 1;
    

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_ddirect_color = dL_ddc; //we removed + idx here
	glm::vec3* dL_dsh = dL_dshs; //we removed + idx here and * max_coeffs

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_ddirect_color[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[0] = dRGBdsh1 * dL_dRGB;
		dL_dsh[1] = dRGBdsh2 * dL_dRGB;
		dL_dsh[2] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[2];
		dRGBdy = -SH_C1 * sh[0];
		dRGBdz = SH_C1 * sh[1];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[3] = dRGBdsh4 * dL_dRGB;
			dL_dsh[4] = dRGBdsh5 * dL_dRGB;
			dL_dsh[5] = dRGBdsh6 * dL_dRGB;
			dL_dsh[6] = dRGBdsh7 * dL_dRGB;
			dL_dsh[7] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[3] + SH_C2[2] * 2.f * -x * sh[5] + SH_C2[3] * z * sh[6] + SH_C2[4] * 2.f * x * sh[7];
			dRGBdy += SH_C2[0] * x * sh[3] + SH_C2[1] * z * sh[4] + SH_C2[2] * 2.f * -y * sh[5] + SH_C2[4] * 2.f * -y * sh[7];
			dRGBdz += SH_C2[1] * y * sh[4] + SH_C2[2] * 2.f * 2.f * z * sh[5] + SH_C2[3] * x * sh[6];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[8] = dRGBdsh9 * dL_dRGB;
				dL_dsh[9] = dRGBdsh10 * dL_dRGB;
				dL_dsh[10] = dRGBdsh11 * dL_dRGB;
				dL_dsh[11] = dRGBdsh12 * dL_dRGB;
				dL_dsh[12] = dRGBdsh13 * dL_dRGB;
				dL_dsh[13] = dRGBdsh14 * dL_dRGB;
				dL_dsh[14] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[8] * 3.f * 2.f * xy +
					SH_C3[1] * sh[9] * yz +
					SH_C3[2] * sh[10] * -2.f * xy +
					SH_C3[3] * sh[11] * -3.f * 2.f * xz +
					SH_C3[4] * sh[12] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[13] * 2.f * xz +
					SH_C3[6] * sh[14] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[8] * 3.f * (xx - yy) +
					SH_C3[1] * sh[9] * xz +
					SH_C3[2] * sh[10] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[11] * -3.f * 2.f * yz +
					SH_C3[4] * sh[12] * -2.f * xy +
					SH_C3[5] * sh[13] * -2.f * yz +
					SH_C3[6] * sh[14] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[9] * xy +
					SH_C3[2] * sh[10] * 4.f * 2.f * yz +
					SH_C3[3] * sh[11] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[12] * 4.f * 2.f * xz +
					SH_C3[5] * sh[13] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[0] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

__device__ void computeCov2DCUDA_device(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* opacities,
	const float* dL_dconics,
	float* dL_dopacity,
	const float* dL_dinvdepth,
	float3* dL_dmeans,
	float* dL_dcov,
	bool antialiasing,
    int idx)
{
	// auto idx = cg::this_grid().thread_rank();
	// if (idx >= P || !(radii[idx] > 0))
	// 	return;
    if(!(radii[idx] > 0)){
        return;
    }

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[0], dL_dconics[1], dL_dconics[3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float c_xx = cov2D[0][0];
	float c_xy = cov2D[0][1];
	float c_yy = cov2D[1][1];
	
	constexpr float h_var = 0.3f;
	float d_inside_root = 0.f;
	if(antialiasing)
	{
		const float det_cov = c_xx * c_yy - c_xy * c_xy;
		c_xx += h_var;
		c_yy += h_var;
		const float det_cov_plus_h_cov = c_xx * c_yy - c_xy * c_xy;
		const float h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability
		const float dL_dopacity_v = dL_dopacity[0];
		const float d_h_convolution_scaling = dL_dopacity_v * opacities[idx];
		dL_dopacity[0] = dL_dopacity_v * h_convolution_scaling;
		d_inside_root = (det_cov / det_cov_plus_h_cov) <= 0.000025f ? 0.f : d_h_convolution_scaling / (2 * h_convolution_scaling);
	} 
	else
	{
		c_xx += h_var;
		c_yy += h_var;
	}
	
	float dL_dc_xx = 0;
	float dL_dc_xy = 0;
	float dL_dc_yy = 0;
	if(antialiasing)
	{
		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdx
		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdz
		const float x = c_xx;
		const float y = c_yy;
		const float z = c_xy;
		const float w = h_var;
		const float denom_f = d_inside_root / sq(w * w + w * (x + y) + x * y - z * z);
		const float dL_dx = w * (w * y + y * y + z * z) * denom_f;
		const float dL_dy = w * (w * x + x * x + z * z) * denom_f;
		const float dL_dz = -2.f * w * z * (w + x + y) * denom_f;
		dL_dc_xx = dL_dx;
		dL_dc_yy = dL_dy;
		dL_dc_xy = dL_dz;
	}
	
	float denom = c_xx * c_yy - c_xy * c_xy;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		
		dL_dc_xx += denom2inv * (-c_yy * c_yy * dL_dconic.x + 2 * c_xy * c_yy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.z);
		dL_dc_yy += denom2inv * (-c_xx * c_xx * dL_dconic.z + 2 * c_xx * c_xy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.x);
		dL_dc_xy += denom2inv * 2 * (c_xy * c_yy * dL_dconic.x - (denom + 2 * c_xy * c_xy) * dL_dconic.y + c_xx * c_xy * dL_dconic.z);
		
		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[0] = (T[0][0] * T[0][0] * dL_dc_xx + T[0][0] * T[1][0] * dL_dc_xy + T[1][0] * T[1][0] * dL_dc_yy);
		dL_dcov[3] = (T[0][1] * T[0][1] * dL_dc_xx + T[0][1] * T[1][1] * dL_dc_xy + T[1][1] * T[1][1] * dL_dc_yy);
		dL_dcov[5] = (T[0][2] * T[0][2] * dL_dc_xx + T[0][2] * T[1][2] * dL_dc_xy + T[1][2] * T[1][2] * dL_dc_yy);
		
		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[1] = 2 * T[0][0] * T[0][1] * dL_dc_xx + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][1] * dL_dc_yy;
		dL_dcov[2] = 2 * T[0][0] * T[0][2] * dL_dc_xx + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][2] * dL_dc_yy;
		dL_dcov[4] = 2 * T[0][2] * T[0][1] * dL_dc_xx + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_dc_xy + 2 * T[1][1] * T[1][2] * dL_dc_yy;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xx +
	(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_xy;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xx +
	(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_xy;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xx +
	(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_xy;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_yy +
	(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xy;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_yy +
	(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xy;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_yy +
	(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xy;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12
		- dL_dinvdepth[0] * tz2; // we removed idx..

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[0] = dL_dmean;
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSHForward(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* dc, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* direct_color = ((glm::vec3*)dc) + idx;
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * direct_color[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[0] + SH_C1 * z * sh[1] - SH_C1 * x * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[3] +
				SH_C2[1] * yz * sh[4] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[5] +
				SH_C2[3] * xz * sh[6] +
				SH_C2[4] * (xx - yy) * sh[7];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[8] +
					SH_C3[1] * xy * z * sh[9] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[10] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[11] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[12] +
					SH_C3[5] * z * (xx - yy) * sh[13] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[14];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[0] = (result.x < 0);
	clamped[1] = (result.y < 0);
	clamped[2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2DForward(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__device__ void computeCov3D_device(
    int idx, 
    const glm::vec3 scale, 
    float mod, 
    const glm::vec4 rot, 
    const float* dL_dcov3Ds, 
    glm::vec3* dL_dscales, 
    glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds;// + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales; // + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
}

// template<int C>
__device__ void preprocessCUDA_device(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* dc,
	const float* shs,
	bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_ddc,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dopacity,
    int idx)
{
	// auto idx = cg::this_grid().thread_rank();
	// if (idx >= P || !(radii[idx] > 0))
	// 	return;

    if(!(radii[idx] > 0)){
        return;
    }

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[0].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[0].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[0].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[0].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[0].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[0].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[0] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH_device(idx, D, M, (glm::vec3*)means, *campos, dc, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_ddc, (glm::vec3*)dL_dsh);

    // glm::vec4 rot = rotations[idx]; // / glm::length(rotations[idx]);
	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D_device(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

template<int C>
__device__
void precompute_gradient_help_variables(
    const uint32_t gaussian_id,
    const uint32_t pix_id,
    const int max_coeffs, // max_coeffs = M  
    int P, 
    int D, 
    const float* means3D,
    const float* dc,
    const float* shs,
    bool* clamped,
    const float* cov3Ds,
    const float* opacities,
    const float* viewmatrix,
    const float* projmatrix,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    const float* campos,
    float2* d,
    float* G,
    float4* con_o,
    glm::vec3* color,
    bool antialiasing
){
    float3 p_orig = { means3D[3 * gaussian_id], means3D[3 * gaussian_id + 1], means3D[3 * gaussian_id + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
    float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

    const float* cov3D = cov3Ds + gaussian_id * 6;
    float3 cov = computeCov2DForward(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	float det = det_cov_plus_h_cov;

	if (det == 0.0f){
		return;
    }
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

    *con_o = { conic.x, conic.y, conic.z, opacities[gaussian_id] * h_convolution_scaling };

    const uint2 pix = {pix_id % W, pix_id / W };
    const float2 pixf = {(float) pix.x, (float) pix.y};
    const float2 xy = point_image;

    *d = { xy.x - pixf.x, xy.y - pixf.y };
    const float power = -0.5f * (con_o->x * d->x * d->x + con_o->z * d->y * d->y) - con_o->y * d->x * d->y;
    // if (power > 0.0f) return;
    *G = exp(power);

    *color = computeColorFromSHForward(gaussian_id, D, max_coeffs, (glm::vec3*)means3D, *(glm::vec3*)campos, dc, shs, clamped);
}

template<int C>
__device__
void compute_gradients(
    float T,
    float* ar,
    int ch,
    uint32_t gaussian_id,
    uint32_t pix_id,
    float G,
    float2 d,
    glm::vec3 color,
    float4 con_o,
    uint32_t N, 
    uint32_t M,
    int P, 
    int D, 
    int num_views,
    int max_coeffs, // max_coeffs = M
    const float* means3D, //const float3* means3D,
    const int* radii,
    const float* dc,
    const float* shs,
    bool* clamped,
    const float* opacities,
    const float* scales, //const glm::vec3* scales,
    const float* rotations, //const glm::vec4* rotations,
    const float scale_modifier,
    const float* cov3Ds,
    const float* viewmatrix,
    const float* projmatrix,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    const float* campos, //const glm::vec3* campos,
    bool antialiasing,
    float3* dr_dmean,
    glm::vec3* dr_dscale,
    glm::vec4* dr_drot,
    float* dr_dopacity,
    glm::vec3* dr_ddc,
    glm::vec3* dr_dsh
){

    const float alpha = min(0.99f, con_o.w * G);

    if (alpha < 1.0f / 255.0f)
		return;

    if (T < 0.0001f)
        return;

    const float weight = alpha * T;

    float bg_dot_dpixel = 0.0f;
    float dr_dcolors[C] = {0.0f, 0.0f, 0.0f};
	float dr_dalpha_channel = 0.0f;

    const float dr_dch = -1.0f;
			
    
    dr_dcolors[ch] = weight * dr_dch;
    dr_dalpha_channel = ((color[ch] * T) - (1.0f / (1.0f - alpha)) * (-ar[ch])) * dr_dch; 
    // printf("G: %d, P: %d, ch: %d, ar: (%g, %g, %g),G: %g, dr_dcolor: (%f, %f, %f), weight:%f\n", gaussian_id, pix_id, ch,  ar[0], ar[1], ar[2], G, dr_dcolors[0], dr_dcolors[1], dr_dcolors[2], weight);
	// printf("pix: (x: %d, y:%d)\n", pix_id % W, pix_id / W);
	
    const float gdx = G * d.x;
    const float gdy = G * d.y;

    const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
    const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

    const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;


    const float dL_dG = con_o.w * dr_dalpha_channel;

    const float dr_dmean2D_x = dL_dG * dG_ddelx * ddelx_dx;
    const float dr_dmean2D_y = dL_dG * dG_ddely * ddely_dy;
    
    const float dr_dconic2D_x= -0.5f * gdx * d.x * dL_dG;
    const float dr_dconic2D_y= -0.5f * gdx * d.y * dL_dG;
    const float dr_dconic2D_w= -0.5f * gdy * d.y * dL_dG;

    const float dr_dinvdepths = 0.0f;

    
    float3 dr_dmean2D = {dr_dmean2D_x, dr_dmean2D_y, 0.0f};
    float dr_dconics[4] = {dr_dconic2D_x, dr_dconic2D_y, 0.0f, dr_dconic2D_w};
    
    float dr_dcov3D[6] = {}; 

    *dr_dopacity = G * dr_dalpha_channel;

    

	// if(pix_id == 265901){
	// 	// printf("d.x, d.y, G: %g, %g, %g, pixel_id: %d, channel: %d: \n", d.x, d.y, G, y, ch);
	// 	printf("pix_id: %d, ch: %d, dr_dmean2D: (%g, %g, %g), G: %g\n", pix_id, ch, dr_dmean2D.x, dr_dmean2D.y, dr_dmean2D.z, G);	
	// }

    computeCov2DCUDA_device(
        P,
        (float3*)means3D,
        radii,
        cov3Ds,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        viewmatrix,
        opacities,
        &dr_dconics[0],
        dr_dopacity,
        &dr_dinvdepths,
        dr_dmean,
        &dr_dcov3D[0],
        antialiasing,
        gaussian_id
    );

	

    preprocessCUDA_device(
        P, D, max_coeffs,
        (float3*)means3D,
        radii,
        dc,
        shs,
        clamped,
        (glm::vec3*)scales,
        (glm::vec4*)rotations,
        scale_modifier,
        projmatrix,
        (glm::vec3*)campos,
        &dr_dmean2D,
        (glm::vec3*)dr_dmean,
        &dr_dcolors[0],
        &dr_dcov3D[0],
        (float*)dr_ddc,
        (float*)dr_dsh,
        dr_dscale,
        dr_drot,
        dr_dopacity,
        gaussian_id
    );

	

    glm::vec3 scale = {scales[gaussian_id*3 + 0], scales[gaussian_id*3 + 1], scales[gaussian_id*3 + 2]};
    dr_dscale->x = dr_dscale->x * scale.x;
    dr_dscale->y = dr_dscale->y * scale.y;
    dr_dscale->z = dr_dscale->z * scale.z;

    float real_opacity_val = log(opacities[gaussian_id]/(1-opacities[gaussian_id]));
    *dr_dopacity = *dr_dopacity * sigmoid_grad(real_opacity_val);

    // float4 rot_original = {rotations[gaussian_id*4+0],rotations[gaussian_id*4+1],rotations[gaussian_id*4+2],rotations[gaussian_id*4+3]};

    // float4 dr_drot_un = dnormvdv(rot_original, float4{dr_drot->x, dr_drot->y, dr_drot->z, dr_drot->w});
    // dr_drot->x = dr_drot_un.x;
    // dr_drot->y = dr_drot_un.y;
    // dr_drot->z = dr_drot_un.z;
    // dr_drot->w = dr_drot_un.w;
}



template<uint32_t C>
__global__
void applyJr(
    double* v, 
    float* residuals, 
    const float* cache, // cached values for gradient calculations (T, ar[3], contributes)
    const uint64_t* cache_indices, //index of values in J: int index = gaussian_idx + pix_id * P + view_index * P * (W*H);
    const uint32_t num_cache_entries, // Number of non 
    uint32_t N, 
    uint32_t M,
    int P, // Num gaussians
    int D, 
    int num_views,
    int max_coeffs, // max_coeffs = M
    const float* means3D, //const float3* means3D,
    const int* radii,
    const float* dc,
    const float* shs,
    const bool* clamped,
    const float* opacities,
    const float* scales, //const glm::vec3* scales,
    const float* rotations, //const glm::vec4* rotations,
    const float scale_modifier,
    const float* cov3Ds,
    const float* viewmatrix,
    const float* projmatrix,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    const float* campos, //const glm::vec3* campos,
    bool antialiasing
){
    uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(t_idx >= num_cache_entries) return;

    const uint64_t index = cache_indices[t_idx];
    const uint32_t gaussian_id = (index % (P*W*H)) % P;
    const uint32_t pix_id = (index % (P*W*H)) / P;
    const uint32_t view_idx = index / (P*W*H);

	

    float T = cache[t_idx * VALUES_PER_CACHE_ENTIRE + 0];
    float ar[C] = {
        cache[t_idx * VALUES_PER_CACHE_ENTIRE + 1],
        cache[t_idx * VALUES_PER_CACHE_ENTIRE + 2],
        cache[t_idx * VALUES_PER_CACHE_ENTIRE + 3],
    };

    // printf("P: %d, ar: (%g, %g, %g), T: %g \n",pix_id,  ar[0], ar[1], ar[2], T);

    float contributor = cache[t_idx * VALUES_PER_CACHE_ENTIRE + 4];

    const float* viewmatrix_ptr = viewmatrix + (16 * view_idx);
    const float* projmatrix_ptr = projmatrix + (16 * view_idx);
    const float* campos_ptr = campos + (3 * view_idx);
    const int* radii_ptr = radii + view_idx;

    float G;
    float2 d;
    glm::vec3 color;
    float4 con_o;

    bool l_clamped[3] = {};

    precompute_gradient_help_variables<C>(
        gaussian_id,
        pix_id,
        max_coeffs,
        P, 
        D, 
        means3D,
        dc,
        shs,
        l_clamped,
        cov3Ds,
        opacities,
        viewmatrix_ptr,
        projmatrix_ptr,
        W, H,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        campos_ptr,
        &d,
        &G,
        &con_o,
        &color,
        antialiasing
    );

    for(int ch = 0; ch < C; ch++){

        float dr_dopacity = 0.0f;
        float3 dr_dmean = {0.0f, 0.0f, 0.0f}; 
        glm::vec4 dr_drot = glm::vec4(0.0f);
        glm::vec3 dr_dscale = glm::vec3(0.0f);
        glm::vec3 dr_ddc = glm::vec3(0.0f);
        float raw_dr_dsh[15*3] = {};
        glm::vec3* dr_dsh = (glm::vec3*)raw_dr_dsh;  

        compute_gradients<C>(
            T,
            ar,
            ch,
            gaussian_id,
            pix_id,
            G,
            d,
            color,
            con_o,
            N, 
            M,
            P, 
            D, 
            num_views,
            max_coeffs, // max_coeffs = M
            means3D, //const float3* means3D,
            radii_ptr,
            dc,
            shs,
            l_clamped,
            opacities,
            scales, //const glm::vec3* scales,
            rotations, //const glm::vec4* rotations,
            scale_modifier,
            cov3Ds,
            viewmatrix_ptr,
            projmatrix_ptr,
            W, H,
            focal_x, focal_y,
            tan_fovx, tan_fovy,
            campos_ptr, //const glm::vec3* campos,
            antialiasing,
            &dr_dmean,
            &dr_dscale,
            &dr_drot,
            &dr_dopacity,
            &dr_ddc,
            dr_dsh
        );

        float residual = residuals[view_idx * C * M + pix_id * C + ch];


        atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 0], -dr_dmean.x * residual);
        atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 1], -dr_dmean.y * residual);
        atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 2], -dr_dmean.z * residual);
    
        atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 3], -dr_dscale.x * residual);
        atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 4], -dr_dscale.y * residual);
        atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 5], -dr_dscale.z * residual);
    
        atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 6], -dr_drot.x * residual);
        atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 7], -dr_drot.y * residual);
        atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 8], -dr_drot.z * residual);
        atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 9], -dr_drot.w * residual);
    
        atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 10], -dr_dopacity * residual);
    
        atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 11], -dr_ddc.x * residual);
        atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 12], -dr_ddc.y * residual);
        atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 13], -dr_ddc.z * residual);
    
        for(int i = 0; i < max_coeffs; i++){
            atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 14 + i * 3 ], -dr_dsh[i].x * residual);
            atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 15 + i * 3 ], -dr_dsh[i].y * residual);
            atomicAdd(&v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 16 + i * 3 ], -dr_dsh[i].z * residual);
        }
    }        

}




template<uint32_t C>
__global__ 
void diagJTJ(
    double* M_precon, 
    const float* cache, // cached values for gradient calculations (T, ar[3], contributes)
    const uint64_t* cache_indices, //index of values in J: int index = gaussian_idx + pix_id * P + view_index * P * (W*H);
    const uint32_t num_cache_entries, // Number of non 
    uint32_t N, 
    uint32_t M,
    int P, // Num gaussians
    int D, 
    int num_views,
    int max_coeffs, // max_coeffs = M
    const float* means3D, //const float3* means3D,
    const int* radii,
    const float* dc,
    const float* shs,
    const bool* clamped,
    const float* opacities,
    const float* scales, //const glm::vec3* scales,
    const float* rotations, //const glm::vec4* rotations,
    const float scale_modifier,
    const float* cov3Ds,
    const float* viewmatrix,
    const float* projmatrix,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    const float* campos, //const glm::vec3* campos,
    bool antialiasing
){
    uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(t_idx >= num_cache_entries) return;

    const uint64_t index = cache_indices[t_idx];
    const uint32_t gaussian_id = (index % (P*W*H)) % P;
    const uint32_t pix_id = (index % (P*W*H)) / P;
    const uint32_t view_idx = index / (P*W*H);

	

    float T = cache[t_idx * VALUES_PER_CACHE_ENTIRE + 0];
    float ar[C] = {
        cache[t_idx * VALUES_PER_CACHE_ENTIRE + 1],
        cache[t_idx * VALUES_PER_CACHE_ENTIRE + 2],
        cache[t_idx * VALUES_PER_CACHE_ENTIRE + 3],
    };

    // printf("P: %d, ar: (%g, %g, %g), T: %g \n",pix_id,  ar[0], ar[1], ar[2], T);

    float contributor = cache[t_idx * VALUES_PER_CACHE_ENTIRE + 4];

    const float* viewmatrix_ptr = viewmatrix + (16 * view_idx);
    const float* projmatrix_ptr = projmatrix + (16 * view_idx);
    const float* campos_ptr = campos + (3 * view_idx);
    const int* radii_ptr = radii + view_idx;

    float G;
    float2 d;
    glm::vec3 color;
    float4 con_o;

    bool l_clamped[3] = {};

    precompute_gradient_help_variables<C>(
        gaussian_id,
        pix_id,
        max_coeffs,
        P, 
        D, 
        means3D,
        dc,
        shs,
        l_clamped,
        cov3Ds,
        opacities,
        viewmatrix_ptr,
        projmatrix_ptr,
        W, H,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        campos_ptr,
        &d,
        &G,
        &con_o,
        &color,
        antialiasing
    );

    for(int ch = 0; ch < C; ch++){

        float dr_dopacity = 0.0f;
        float3 dr_dmean = {0.0f, 0.0f, 0.0f}; 
        glm::vec4 dr_drot = glm::vec4(0.0f);
        glm::vec3 dr_dscale = glm::vec3(0.0f);
        glm::vec3 dr_ddc = glm::vec3(0.0f);
        float raw_dr_dsh[15*3] = {};
        glm::vec3* dr_dsh = (glm::vec3*)raw_dr_dsh;  

        compute_gradients<C>(
            T,
            ar,
            ch,
            gaussian_id,
            pix_id,
            G,
            d,
            color,
            con_o,
            N, 
            M,
            P, 
            D, 
            num_views,
            max_coeffs, // max_coeffs = M
            means3D, //const float3* means3D,
            radii_ptr,
            dc,
            shs,
            l_clamped,
            opacities,
            scales, //const glm::vec3* scales,
            rotations, //const glm::vec4* rotations,
            scale_modifier,
            cov3Ds,
            viewmatrix_ptr,
            projmatrix_ptr,
            W, H,
            focal_x, focal_y,
            tan_fovx, tan_fovy,
            campos_ptr, //const glm::vec3* campos,
            antialiasing,
            &dr_dmean,
            &dr_dscale,
            &dr_drot,
            &dr_dopacity,
            &dr_ddc,
            dr_dsh
        );


		atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 0], dr_dmean.x * dr_dmean.x);
        atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 1], dr_dmean.y * dr_dmean.y);
        atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 2], dr_dmean.z * dr_dmean.z);

        atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 3], dr_dscale.x * dr_dscale.x);
        atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 4], dr_dscale.y * dr_dscale.y);
        atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 5], dr_dscale.z * dr_dscale.z);

        atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 6], dr_drot.x * dr_drot.x);
        atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 7], dr_drot.y * dr_drot.y);
        atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 8], dr_drot.z * dr_drot.z);
        atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 9], dr_drot.w * dr_drot.w);

        atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 10], dr_dopacity * dr_dopacity);

        atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 11], dr_ddc.x * dr_ddc.x);
        atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 12], dr_ddc.y * dr_ddc.y);
        atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 13], dr_ddc.z * dr_ddc.z);

        for(int i = 0; i < max_coeffs; i++){
            atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 14 + i * 3], dr_dsh[i].x * dr_dsh[i].x);
            atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 15 + i * 3], dr_dsh[i].y * dr_dsh[i].y);
            atomicAdd(&M_precon[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 16 + i * 3], dr_dsh[i].z * dr_dsh[i].z);
        }
	}
}

template<uint32_t C>
__global__
void residual_dot_sum(
	double* v, 
	double* residual_dot_v, 
    const float* cache, // cached values for gradient calculations (T, ar[3], contributes)
    const uint64_t* cache_indices, //index of values in J: int index = gaussian_idx + pix_id * P + view_index * P * (W*H);
    const uint32_t num_cache_entries, // Number of non 
    uint32_t N, 
    uint32_t M,
    int P, // Num gaussians
    int D, 
    int num_views,
    int max_coeffs, // max_coeffs = M
    const float* means3D, //const float3* means3D,
    const int* radii,
    const float* dc,
    const float* shs,
    const bool* clamped,
    const float* opacities,
    const float* scales, //const glm::vec3* scales,
    const float* rotations, //const glm::vec4* rotations,
    const float scale_modifier,
    const float* cov3Ds,
    const float* viewmatrix,
    const float* projmatrix,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    const float* campos, //const glm::vec3* campos,
    bool antialiasing
){
    uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(t_idx >= num_cache_entries) return;

    const uint64_t index = cache_indices[t_idx];
    const uint32_t gaussian_id = (index % (P*W*H)) % P;
    const uint32_t pix_id = (index % (P*W*H)) / P;
    const uint32_t view_idx = index / (P*W*H);

	

    float T = cache[t_idx * VALUES_PER_CACHE_ENTIRE + 0];
    float ar[C] = {
        cache[t_idx * VALUES_PER_CACHE_ENTIRE + 1],
        cache[t_idx * VALUES_PER_CACHE_ENTIRE + 2],
        cache[t_idx * VALUES_PER_CACHE_ENTIRE + 3],
    };

    // printf("P: %d, ar: (%g, %g, %g), T: %g \n",pix_id,  ar[0], ar[1], ar[2], T);

    float contributor = cache[t_idx * VALUES_PER_CACHE_ENTIRE + 4];

    const float* viewmatrix_ptr = viewmatrix + (16 * view_idx);
    const float* projmatrix_ptr = projmatrix + (16 * view_idx);
    const float* campos_ptr = campos + (3 * view_idx);
    const int* radii_ptr = radii + view_idx;

    float G;
    float2 d;
    glm::vec3 color;
    float4 con_o;

    bool l_clamped[3] = {};

    precompute_gradient_help_variables<C>(
        gaussian_id,
        pix_id,
        max_coeffs,
        P, 
        D, 
        means3D,
        dc,
        shs,
        l_clamped,
        cov3Ds,
        opacities,
        viewmatrix_ptr,
        projmatrix_ptr,
        W, H,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        campos_ptr,
        &d,
        &G,
        &con_o,
        &color,
        antialiasing
    );

    for(int ch = 0; ch < C; ch++){

        float dr_dopacity = 0.0f;
        float3 dr_dmean = {0.0f, 0.0f, 0.0f}; 
        glm::vec4 dr_drot = glm::vec4(0.0f);
        glm::vec3 dr_dscale = glm::vec3(0.0f);
        glm::vec3 dr_ddc = glm::vec3(0.0f);
        float raw_dr_dsh[15*3] = {};
        glm::vec3* dr_dsh = (glm::vec3*)raw_dr_dsh;  

        compute_gradients<C>(
            T,
            ar,
            ch,
            gaussian_id,
            pix_id,
            G,
            d,
            color,
            con_o,
            N, 
            M,
            P, 
            D, 
            num_views,
            max_coeffs, // max_coeffs = M
            means3D, //const float3* means3D,
            radii_ptr,
            dc,
            shs,
            l_clamped,
            opacities,
            scales, //const glm::vec3* scales,
            rotations, //const glm::vec4* rotations,
            scale_modifier,
            cov3Ds,
            viewmatrix_ptr,
            projmatrix_ptr,
            W, H,
            focal_x, focal_y,
            tan_fovx, tan_fovy,
            campos_ptr, //const glm::vec3* campos,
            antialiasing,
            &dr_dmean,
            &dr_dscale,
            &dr_drot,
            &dr_dopacity,
            &dr_ddc,
            dr_dsh
        );

		atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_dmean.x * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 0]);
        atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_dmean.y * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 1]);
        atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_dmean.z * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 2]);

        atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_dscale.x * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 3]);
        atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_dscale.y * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 4]);
        atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_dscale.z * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 5]);

        atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_drot.x * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 6]);
        atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_drot.y * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 7]);
        atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_drot.z * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 8]);
        atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_drot.w * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 9]);

        atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_dopacity * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 10]);

        atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_ddc.x * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 11]);
        atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_ddc.y * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 12]);
        atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_ddc.z * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 13]);

        for(int i = 0; i < max_coeffs; i++){
            atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_dsh[i].x * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 14 + i * 3]);
            atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_dsh[i].y * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 15 + i * 3]);
            atomicAdd(&residual_dot_v[view_idx * C * M + pix_id * C + ch], dr_dsh[i].z * v[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 16 + i * 3]);
        }
	}
}

template<uint32_t C>
__global__
void sum_residuals(
    double* Av, 
	double* residual_dot_v, 
    const float* cache, // cached values for gradient calculations (T, ar[3], contributes)
    const uint64_t* cache_indices, //index of values in J: int index = gaussian_idx + pix_id * P + view_index * P * (W*H);
    const uint32_t num_cache_entries, // Number of non 
    uint32_t N, 
    uint32_t M,
    int P, // Num gaussians
    int D, 
    int num_views,
    int max_coeffs, // max_coeffs = M
    const float* means3D, //const float3* means3D,
    const int* radii,
    const float* dc,
    const float* shs,
    const bool* clamped,
    const float* opacities,
    const float* scales, //const glm::vec3* scales,
    const float* rotations, //const glm::vec4* rotations,
    const float scale_modifier,
    const float* cov3Ds,
    const float* viewmatrix,
    const float* projmatrix,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    const float* campos, //const glm::vec3* campos,
    bool antialiasing
){
    uint32_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(t_idx >= num_cache_entries) return;

    const uint64_t index = cache_indices[t_idx];
    const uint32_t gaussian_id = (index % (P*W*H)) % P;
    const uint32_t pix_id = (index % (P*W*H)) / P;
    const uint32_t view_idx = index / (P*W*H);

	

    float T = cache[t_idx * VALUES_PER_CACHE_ENTIRE + 0];
    float ar[C] = {
        cache[t_idx * VALUES_PER_CACHE_ENTIRE + 1],
        cache[t_idx * VALUES_PER_CACHE_ENTIRE + 2],
        cache[t_idx * VALUES_PER_CACHE_ENTIRE + 3],
    };

    // printf("P: %d, ar: (%g, %g, %g), T: %g \n",pix_id,  ar[0], ar[1], ar[2], T);

    float contributor = cache[t_idx * VALUES_PER_CACHE_ENTIRE + 4];

    const float* viewmatrix_ptr = viewmatrix + (16 * view_idx);
    const float* projmatrix_ptr = projmatrix + (16 * view_idx);
    const float* campos_ptr = campos + (3 * view_idx);
    const int* radii_ptr = radii + view_idx;

    float G;
    float2 d;
    glm::vec3 color;
    float4 con_o;

    bool l_clamped[3] = {};

    precompute_gradient_help_variables<C>(
        gaussian_id,
        pix_id,
        max_coeffs,
        P, 
        D, 
        means3D,
        dc,
        shs,
        l_clamped,
        cov3Ds,
        opacities,
        viewmatrix_ptr,
        projmatrix_ptr,
        W, H,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        campos_ptr,
        &d,
        &G,
        &con_o,
        &color,
        antialiasing
    );

    for(int ch = 0; ch < C; ch++){

        float dr_dopacity = 0.0f;
        float3 dr_dmean = {0.0f, 0.0f, 0.0f}; 
        glm::vec4 dr_drot = glm::vec4(0.0f);
        glm::vec3 dr_dscale = glm::vec3(0.0f);
        glm::vec3 dr_ddc = glm::vec3(0.0f);
        float raw_dr_dsh[15*3] = {};
        glm::vec3* dr_dsh = (glm::vec3*)raw_dr_dsh;  

        compute_gradients<C>(
            T,
            ar,
            ch,
            gaussian_id,
            pix_id,
            G,
            d,
            color,
            con_o,
            N, 
            M,
            P, 
            D, 
            num_views,
            max_coeffs, // max_coeffs = M
            means3D, //const float3* means3D,
            radii_ptr,
            dc,
            shs,
            l_clamped,
            opacities,
            scales, //const glm::vec3* scales,
            rotations, //const glm::vec4* rotations,
            scale_modifier,
            cov3Ds,
            viewmatrix_ptr,
            projmatrix_ptr,
            W, H,
            focal_x, focal_y,
            tan_fovx, tan_fovy,
            campos_ptr, //const glm::vec3* campos,
            antialiasing,
            &dr_dmean,
            &dr_dscale,
            &dr_drot,
            &dr_dopacity,
            &dr_ddc,
            dr_dsh
        );

		atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 0], dr_dmean.x * residual_dot_v[view_idx * C * M + pix_id * C + ch]);
        atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 1], dr_dmean.y * residual_dot_v[view_idx * C * M + pix_id * C + ch]);
        atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 2], dr_dmean.z * residual_dot_v[view_idx * C * M + pix_id * C + ch]);

        atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 3], dr_dscale.x * residual_dot_v[view_idx * C * M + pix_id * C + ch]);
        atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 4], dr_dscale.y * residual_dot_v[view_idx * C * M + pix_id * C + ch]);
        atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 5], dr_dscale.z * residual_dot_v[view_idx * C * M + pix_id * C + ch]);

        atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 6], dr_drot.x * residual_dot_v[view_idx * C * M + pix_id * C + ch]);
        atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 7], dr_drot.y * residual_dot_v[view_idx * C * M + pix_id * C + ch]);
        atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 8], dr_drot.z * residual_dot_v[view_idx * C * M + pix_id * C + ch]);
        atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 9], dr_drot.w * residual_dot_v[view_idx * C * M + pix_id * C + ch]);

        atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 10], dr_dopacity * residual_dot_v[view_idx * C * M + pix_id * C + ch]);

        atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 11], dr_ddc.x * residual_dot_v[view_idx * C * M + pix_id * C + ch]);
        atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 12], dr_ddc.y * residual_dot_v[view_idx * C * M + pix_id * C + ch]);
        atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 13], dr_ddc.z * residual_dot_v[view_idx * C * M + pix_id * C + ch]);

        for(int i = 0; i < max_coeffs; i++){
            atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 14 + i * 3], dr_dsh[i].x * residual_dot_v[view_idx * C * M + pix_id * C + ch]);
            atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 15 + i * 3], dr_dsh[i].y * residual_dot_v[view_idx * C * M + pix_id * C + ch]);
            atomicAdd(&Av[gaussian_id * NUM_PARAMS_PER_GAUSSIAN + 16 + i * 3], dr_dsh[i].z * residual_dot_v[view_idx * C * M + pix_id * C + ch]);
        }
	}
}

template<uint32_t C>
void Av(
	double* v, 
    double* Av, 
	const float* cache, // cached values for gradient calculations (T, ar[3], contributes)
    const uint64_t* cache_indices, //index of values in J: int index = gaussian_idx + pix_id * P + view_index * P * (W*H);
    const uint32_t num_cache_entries, // Number of non 
	uint32_t N, 
    uint32_t M,
    int P, 
    int D, 
    int num_views,
    int max_coeffs, // max_coeffs = M
    const float* means3D, //const float3* means3D,
    const int* radii,
    const float* dc,
    const float* shs,
    const bool* clamped,
    const float* opacities,
    const float* scales, //const glm::vec3* scales,
    const float* rotations, //const glm::vec4* rotations,
    const float scale_modifier,
    const float* cov3Ds,
    const float* viewmatrix,
    const float* projmatrix,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    const float* campos, //const glm::vec3* campos,
    bool antialiasing
){
	double* residual_dot_v;
    cudaMalloc(&residual_dot_v, num_views * M * C * sizeof(double));
    cudaMemset(residual_dot_v, 0, num_views * M * C * sizeof(double));

	residual_dot_sum<C><<<(num_cache_entries+255)/256, 256>>>(
        v, 
		residual_dot_v,
        cache, // cached values for gradient calculations (T, ar[3], contributes)
        cache_indices, //index of values in J: int index = gaussian_idx + pix_id * P + view_index * P * (W*H);
        num_cache_entries, // Number of non 
        N, 
        M,
        P, // Num gaussians
        D, 
        num_views,
        max_coeffs, // max_coeffs = M
        means3D, //const float3* means3D,
        radii,
        dc,
        shs,
        clamped,
        opacities,
        scales, //const glm::vec3* scales,
        rotations, //const glm::vec4* rotations,
        scale_modifier,
        cov3Ds,
        viewmatrix,
        projmatrix,
        W, H,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        campos, //const glm::vec3* campos,
        antialiasing
    );

	sum_residuals<C><<<(num_cache_entries+255)/256, 256>>>(
        Av, 
		residual_dot_v,
        cache, // cached values for gradient calculations (T, ar[3], contributes)
        cache_indices, //index of values in J: int index = gaussian_idx + pix_id * P + view_index * P * (W*H);
        num_cache_entries, // Number of non 
        N, 
        M,
        P, // Num gaussians
        D, 
        num_views,
        max_coeffs, // max_coeffs = M
        means3D, //const float3* means3D,
        radii,
        dc,
        shs,
        clamped,
        opacities,
        scales, //const glm::vec3* scales,
        rotations, //const glm::vec4* rotations,
        scale_modifier,
        cov3Ds,
        viewmatrix,
        projmatrix,
        W, H,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        campos, //const glm::vec3* campos,
        antialiasing
    );

	cudaFree(residual_dot_v);
}


const char* get_param_name(int index){
    static std::string res = "";
    if(index <= 2) res = "dr_dmean[" + std::to_string(index) + "]";
    else if(index <= 5) res = "dr_dscale[" + std::to_string(index-3) + "]";
    else if(index <= 9) res = "dr_drot[" + std::to_string(index-6) + "]";
    else if(index <= 10) res = "dr_dopacity[" + std::to_string(index-10) + "]";
    else if(index <= 13) res = "dr_ddc[" + std::to_string(index-11) + "]";
    else res = "dr_dsh[" + std::to_string((index-14)%3) + "]";
    return res.c_str();
}


void GaussNewton::gaussNewtonUpdate(
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
    double* preconditioner,
    const float* cache, // cached values for gradient calculations (T, ar[3], contributes)
    const uint64_t* cache_indices,
    const uint32_t num_cache_entries, // Number of non zero values
    float* residuals,
    const bool* tiles_touched,
    const uint32_t N, // number of parameters
    const uint32_t M,  // number of residuals
    const uint32_t num_views, //Number of views that fit in memory
    bool debug
){

    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    printf("tot num params: %d \n", N);
    printf("tot num gaussians: %d \n", P);
    printf("chached residuals: %d \n", num_cache_entries);
    printf("tot num residuals: %d \n", M);
    printf("tot num views: %d \n", num_views);

    double* b;
    cudaMalloc(&b, N * sizeof(double));
    cudaMemset(b,0, N * sizeof(double));

    bool* test_bool;
    cudaMalloc(&test_bool, sizeof(bool));
    cudaMemset(test_bool, 0, sizeof(bool));
    bool h_bool;

    cudaMemset(test_bool, 0, sizeof(bool));
    isnan_test_f<<<(M+255)/256, 256>>>(cache, test_bool, num_cache_entries);
    cudaMemcpy(&h_bool, test_bool, sizeof(bool), cudaMemcpyDeviceToHost);
    printf("cache nan?: %d\n", h_bool);


    cudaMemset(test_bool, 0, sizeof(bool));
    isnan_test_f<<<(M+255)/256, 256>>>(residuals, test_bool, M);
    cudaMemcpy(&h_bool, test_bool, sizeof(bool), cudaMemcpyDeviceToHost);
    printf("residuals nan?: %d\n", h_bool);

    applyJr<NUM_CHANNELS_3DGS><<<(num_cache_entries+255)/256, 256>>>(
        b, 
        residuals, 
        cache, // cached values for gradient calculations (T, ar[3], contributes)
        cache_indices, //index of values in J: int index = gaussian_idx + pix_id * P + view_index * P * (W*H);
        num_cache_entries, // Number of non 
        N, 
        M,
        P, // Num gaussians
        D, 
        num_views,
        max_coeffs, // max_coeffs = M
        means3D, //const float3* means3D,
        radii,
        dc,
        shs,
        clamped,
        opacities,
        scales, //const glm::vec3* scales,
        rotations, //const glm::vec4* rotations,
        scale_modifier,
        cov3Ds,
        viewmatrix,
        projmatrix,
        width, height,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        campos, //const glm::vec3* campos,
        antialiasing
    );

    cudaMemset(test_bool, 0, sizeof(bool));
    isnan_test_d<<<(N+255)/256, 256>>>(b, test_bool, N);
    cudaMemcpy(&h_bool, test_bool, sizeof(bool), cudaMemcpyDeviceToHost);
    printf("b nan?: %d\n", h_bool);

    // double h_float;
    // for(int i = 0; i < N; i++){
    //     cudaMemcpy(&h_float, &b[i], sizeof(double), cudaMemcpyDeviceToHost);
    //     printf("b(%d): %g  | %s \n",i, h_float, get_param_name(i%59));
    //     if(i % 59 == 11){
    //         i = 59 * (i/59) + 58;
    //         printf("\n");
    //     }
    // }

	double* check_b;
    double h_check_b;
    cudaMalloc(&check_b, sizeof(double));
    cudaMemset(check_b, 0, sizeof(double));
    dot_d<<<(N+255)/256, 256>>>(b, b, check_b, N);
    cudaMemcpy(&h_check_b, check_b, sizeof(double), cudaMemcpyDeviceToHost);
    if(h_check_b == 0){
        printf("PCG: no data");
        cudaFree(b);
        return;
    }

	double* Ap;
    cudaMalloc(&Ap, N * sizeof(double));
    cudaMemset(Ap, 0, N * sizeof(double));

    // CHECK_CUDA(Av<NUM_CHANNELS_3DGS>(
    //     x,
    //     Ap,
	// 	cache, 
	// 	cache_indices,
	// 	num_cache_entries,
    //     N,
    //     M,
    //     P, 
    //     D, 
    //     num_views,
    //     max_coeffs, 
    //     means3D, 
    //     radii, 
    //     dc, 
    //     shs, 
    //     clamped, 
    //     opacities, 
    //     scales, 
    //     rotations, 
    //     scale_modifier, 
    //     cov3Ds, 
    //     viewmatrix, 
    //     projmatrix, 
    //     width,
    //     height,
    //     focal_x, 
    //     focal_y, 
    //     tan_fovx, 
    //     tan_fovy, 
    //     campos, 
    //     antialiasing
    // ), debug);

    cudaMemset(test_bool, 0, sizeof(bool));
    isnan_test_d<<<(N+255)/256, 256>>>(Ap, test_bool, N);
    cudaMemcpy(&h_bool, test_bool, sizeof(bool), cudaMemcpyDeviceToHost);
    printf("Ax0 nan?: %d\n", h_bool);

	// r(0) = b - A*x(0)
    double* r;
    cudaMalloc(&r, N * sizeof(double));
    cudaMemset(r, 0, N * sizeof(double));
    subtract_d<<<(N+255)/256, 256>>>(b, Ap, r, N);

    cudaMemset(test_bool, 0, sizeof(bool));
    isnan_test_d<<<(N+255)/256, 256>>>(r, test_bool, N);
    cudaMemcpy(&h_bool, test_bool, sizeof(bool), cudaMemcpyDeviceToHost);
    printf("r nan?: %d\n", h_bool);


    // calculate M = diag(J^T * J)
	//double* preconditioner;
    //cudaMalloc(&M_precon, N * sizeof(double));
    //cudaMemset(M_precon, 0, N * sizeof(double));
    diagJTJ<NUM_CHANNELS_3DGS><<<(num_cache_entries+255)/256, 256>>>(
        preconditioner, 
		cache, 
		cache_indices,
		num_cache_entries,
        N, 
        M,
        P, 
        D, 
        num_views,
        max_coeffs, 
        means3D, 
        radii, 
        dc, 
        shs, 
        clamped, 
        opacities, 
        scales, 
        rotations, 
        scale_modifier, 
        cov3Ds, 
        viewmatrix, 
        projmatrix, 
        width,
        height,
        focal_x, 
        focal_y, 
        tan_fovx, 
        tan_fovy, 
        campos, 
        antialiasing
    );

	// z(0) = M^-1 * r(0)
    double* z;
    cudaMalloc(&z, N * sizeof(double));
    next_z_d<<<(N+255)/256, 256>>>(preconditioner, r, z, N);


    // p(0) = z(0)
    double* p;
    cudaMalloc(&p, N * sizeof(double));
    cudaMemcpy(p, z, N * sizeof(double), cudaMemcpyDeviceToDevice);

    
    float eps = 1e-8;
    double h_R;
    double h_R_prev;
    double* R;
    double* R_prev;
    
    cudaMalloc(&R, sizeof(double));
    cudaMalloc(&R_prev, sizeof(double));
    cudaMemset(R, 0, sizeof(double));
    cudaMemset(R_prev, 0, sizeof(double));

	// R_prev = r(0)^T * r(0)
    dot_d<<<(N+255)/256, 256>>>(r, r, R_prev, N);
    
    // float* alpha;
    // float* numerator;
    // float* denominator;
    // float* beta;
    // cudaMalloc(&alpha, sizeof(float));
    // cudaMalloc(&numerator, sizeof(float));
    // cudaMalloc(&denominator, sizeof(float));
    // cudaMalloc(&beta, sizeof(float));

    double* alpha;
    double* numerator;
    double* denominator;
    double* beta;
    cudaMalloc(&alpha, sizeof(double));
    cudaMalloc(&numerator, sizeof(double));
    cudaMalloc(&denominator, sizeof(double));
    cudaMalloc(&beta, sizeof(double));

    


    double test_d;
    
    int k = 0; 
    const int MAX_ITERATIONS = 10;
    cudaMemcpy(&h_R_prev, R_prev, sizeof(double), cudaMemcpyDeviceToHost);

	while(k < MAX_ITERATIONS){

        // cudaMemset(alpha, 0, sizeof(float));
        // cudaMemset(denominator, 0, sizeof(float));
        // cudaMemset(numerator, 0, sizeof(float));
        // cudaMemset(beta, 0, sizeof(float));
        cudaMemset(alpha, 0, sizeof(double));
        cudaMemset(denominator, 0, sizeof(double));
        cudaMemset(numerator, 0, sizeof(double));
        cudaMemset(beta, 0, sizeof(double));
        cudaMemset(Ap, 0, N * sizeof(double));
        
		// r(k)^T * z(k)
        // dot<<<(N+255)/256, 256>>>(r, z, numerator, N);
        dot_d<<<(N+255)/256, 256>>>(r, z, numerator, N);
        cudaMemcpy(&test_d, numerator, sizeof(double), cudaMemcpyDeviceToHost);
        printf("numerator: %g\n", test_d);


        
        // isnan_test<<<(N+255)/256, 256>>>(r, N, test_bool);
        // bool h_bool;
        // cudaMemcpy(&h_bool, test_bool, sizeof(bool), cudaMemcpyDeviceToHost);
        // printf("r nan?: %d\n", h_bool);
        
        // A*p(k)
		CHECK_CUDA(Av<NUM_CHANNELS_3DGS>(
            p,
			Ap,
			cache, 
			cache_indices,
			num_cache_entries,
			N,
			M,
			P, 
			D, 
			num_views,
			max_coeffs, 
			means3D, 
			radii, 
			dc, 
			shs, 
			clamped, 
			opacities, 
			scales, 
			rotations, 
			scale_modifier, 
			cov3Ds, 
			viewmatrix, 
			projmatrix, 
			width,
			height,
			focal_x, 
			focal_y, 
			tan_fovx, 
			tan_fovy, 
			campos, 
			antialiasing
		),debug);

        cudaMemset(test_bool, 0, sizeof(bool));
        isnan_test_d<<<(N+255)/256, 256>>>(Ap, test_bool, N);
        cudaMemcpy(&h_bool, test_bool, sizeof(bool), cudaMemcpyDeviceToHost);
        printf("Ap nan?: %d\n", h_bool);
        
		// p(k)^T * Ap(k) 
        // dot<<<(N+255)/256, 256>>>(p, Ap, denominator, N);
        dot_d<<<(N+255)/256, 256>>>(p, Ap, denominator, N);
        cudaMemcpy(&test_d, denominator, sizeof(double), cudaMemcpyDeviceToHost);
        printf("denominator: %g\n", test_d);
        
        // alpha = r(k)^T * z(k) / p(k)^T * Ap(k) 
        // scalar_divide<<<1, 1>>>(numerator, denominator, alpha); 
        scalar_divide_d<<<1, 1>>>(numerator, denominator, alpha); 
        cudaMemcpy(&test_d, alpha, sizeof(double), cudaMemcpyDeviceToHost);
        printf("alpha: %g\n", test_d);
        
        
        // x(k+1) = x + alpha * p
        // next_x<<<(N+255)/256, 256>>>(x, p, alpha, N);
        next_x_d<<<(N+255)/256, 256>>>(x, p, alpha, N);


        // r(k)^T * z(k)  (used for beta calculation)
        // cudaMemset(denominator, 0, sizeof(float));
        cudaMemset(denominator, 0, sizeof(double));
        // dot<<<(N+255)/256, 256>>>(r, z, denominator, N); 
        dot_d<<<(N+255)/256, 256>>>(r, z, denominator, N); 


        // r(k+1) = r(k) - alpha * Ap(k)
        // next_r<<<(N+255)/256, 256>>>(r, Ap, alpha, N);
        next_r_d<<<(N+255)/256, 256>>>(r, Ap, alpha, N);

        // R = r(k+1)^T * r(k+1)
        cudaMemset(R, 0, sizeof(double));
        dot_d<<<(N+255)/256, 256>>>(r, r, R, N); 

		// Check if R/Rprev > 0.85 or R < eps
        cudaMemcpy(&h_R, R, sizeof(double), cudaMemcpyDeviceToHost);
        printf("R_prev: %g, R: %g\n", h_R_prev, h_R);
		printf("R/R_prev: %g \n", (h_R/ h_R_prev));
        if (h_R < eps || h_R/ h_R_prev > 0.85f){
            break;
        }

		h_R_prev = h_R;

        // z(k+1) = M^-1 * r(k)
        next_z_d<<<(N+255)/256, 256>>>(preconditioner, r, z, N);
        
        // r(k+1)^T * z(k+1)
        // cudaMemset(numerator, 0, sizeof(float));
        cudaMemset(numerator, 0, sizeof(double));
        // dot<<<(N+255)/256, 256>>>(r, z, numerator, N); 
        dot_d<<<(N+255)/256, 256>>>(r, z, numerator, N); 
        
        // beta = r(k+1)^T * z(k+1) / r(k)^T * z(k)
        // scalar_divide<<<1, 1>>>(numerator, denominator, beta); 
        scalar_divide_d<<<1, 1>>>(numerator, denominator, beta); 

        // p(k+1) = z(k) + beta * p(k)
        // next_p<<<(N+255)/256, 256>>>(z, p, beta, N);
        next_p_d<<<(N+255)/256, 256>>>(z, p, beta, N);


        k++;
	}

    cudaFree(R);
    cudaFree(R_prev);
    cudaFree(numerator);
    cudaFree(denominator);
    cudaFree(alpha);
    cudaFree(beta);
    //cudaFree(M_precon);
    cudaFree(b);
    cudaFree(r);
    cudaFree(z);
    cudaFree(p);
    cudaFree(Ap);
    
    cudaFree(test_bool);
}