#include "auxiliary.h"
#include "gauss_newton.h"
// #include "backward.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#include <string>

#define THREADS_PER_BLOCK 256



__global__
void createJ(){

}

__global__ void diagJTJ(const float* J, float* M_precon, uint32_t N, uint32_t M){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N*M) return;
    uint32_t x = idx % N;
    uint32_t y = idx / N;


    float j_val = J[x + y * N];

    float tmp = j_val * j_val;

    atomicAdd(&M_precon[x],tmp);
}




__global__ void dot(float *a, float *b, float *c, uint32_t N){
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

__global__ void dot_d(float *a, float *b, double *c, uint32_t N){
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
void gpu_copy(float* src, float* dest, const uint32_t N){
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= N) return;
    
    dest[idx] = src[idx];
}
__global__ 
void Scale(float* v, float* s, float* v_acc, uint32_t N){                // Used for calculating Ap in PCG
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= N) return;
    v_acc[idx] = *s * v[idx];
}


__global__
void scalar_divide(const float* numerator, const float* denominator, float* quotient){
    
    if (*denominator != 0.0f) {
        *quotient = *numerator / *denominator;
    } else {
        *quotient = 0.0f;  // Handle division by zero gracefully
    }
}

__global__
void scalar_divide_double(const double* numerator, const double* denominator, float* quotient){
    
    if (*denominator != 0.0f) {
        *quotient = *numerator / *denominator;
    } else {
        *quotient = 0.0f;  // Handle division by zero gracefully
    }
}

__global__
void next_x(float* x, float* p, float* alpha, uint32_t N){
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
void next_r(float* r, float* Ap, float* alpha, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    float new_val = r[idx] - *alpha * Ap[idx];
    r[idx] = new_val;
} 

__global__ void next_z(float* M_precon, float* r, float* z, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    float denominator = M_precon[idx];
    // float inv = 1.0 / denominator;
    // if (denominator == 0.0f){
    //     inv = 1.0;
    // }
    float inv = 1.0 / (denominator + 1e-8f);
    float z_val = inv * r[idx];
    z[idx] = z_val;
}

void PCG(float* J, float gamma, float alpha){
    // Move PCG algo here from GaussNewton below.
    
    
    
}

__global__
void residual_dot_sum(float* J, float* v, float* residual_dot_v, uint32_t N, uint32_t M){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N*M) return;
    uint32_t x = idx % N; // residual
    uint32_t y = idx / N; // parameter

    atomicAdd(&residual_dot_v[y], J[x + y * N] * v[x]);
}


__global__
void sum_residuals(float* Ap, float* residual_dot_p, float* J, uint32_t N, uint32_t M){  // Sums over r(i) with scalar vector product r(i)^T * p
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N*M) return;
    uint32_t x = idx % N;
    uint32_t y = idx / N;
    atomicAdd(&Ap[x], residual_dot_p[y] * J[x + y * N]);
}


void Av(float* J, float* v, float* Av, uint32_t N, uint32_t M){
    float* residual_dot_v;
    cudaMalloc(&residual_dot_v, M * sizeof(float));
    residual_dot_sum<<<((N*M)+255)/256, 256>>>(J, v, residual_dot_v, N, M);    // (r(i)^T * p)
    sum_residuals<<<((N*M)+255)/256, 256>>>(Av, residual_dot_v, J, N, M);    // Ap = r(i) * (r(i)^T * p)
    cudaFree(residual_dot_v);
}




__global__ 
void subtract(float* a, float* b, float* c, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    c[idx] = a[idx] - b[idx];
}

__global__ 
void sumContrib(int W, int H, const uint32_t* n_contrib, int* sum){
    uint2 pix = { blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
    uint32_t pix_id = W * pix.y + pix.x;
    int val = n_contrib[pix_id];
    atomicAdd(sum, val);

}



__global__
void JTv(float* out_JTv, float* sparse_J, uint64_t* indices, float* v, uint32_t N, uint32_t M, uint32_t num_entries){

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t residual_id = indices[idx] / M;
    uint64_t gaussian_id = indices[idx] % M;

    // execute once per gaussian parameter
    // TODO: do extra calculations for each
    float result = sparse_J[idx] * v[residual_id];

    atomicAdd(&out_JTv[gaussian_id], result);
}


// __device__ 
// float sigmoid(float x){
//     float s = 1 / (1-expf(x));
//     return s;
// }

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

	glm::vec3* direct_color = ((glm::vec3*)dc);  //we removed + idx here
	glm::vec3* sh = ((glm::vec3*)shs); //we removed + idx here and * max_coeffs

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[0];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

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
		- dL_dinvdepth[idx] * tz2;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[0] = dL_dmean;
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

	const float* dL_dcov3D = dL_dcov3Ds; // removed + 6 * idx;

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
	glm::vec3* dL_dscale = dL_dscales; // removed + idx;
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
	float4* dL_drot = (float4*)(dL_drots); // removed + idx;
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

template<int C>
__device__ void preprocessCUDA_device(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* dc,
	const float* shs,
	const bool* clamped,
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

    glm::vec4 rot = rotations[idx] / glm::length(rotations[idx]);
	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D_device(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}



template<uint32_t C>
__global__
void calc_b_non_sparse(
    float* b, 
    float* J,
    float* loss_residuals, 
    uint32_t N, 
    uint32_t M,
    int P, 
    int D, 
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
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    const float* campos, //const glm::vec3* campos,
    bool antialiasing
    ){

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= P*M) return;

    int x = idx % P;  // Gaussian index
    int y = idx / P;  // Pixel index

    // Calculate the residuals wrt parameters
    int index= x + y * P;
    float dL_dmean2D_x = J[index*10];
	float dL_dmean2D_y = J[index*10 + 1];
	float dL_dconic2D_x = J[index*10 + 2] ;
	float dconic2D_y = J[index*10 + 3];
	float dL_dconic2D_w = J[index*10 + 4];
    float dL_dconics[4] = {dL_dconic2D_x, dconic2D_y, 0.0f, dL_dconic2D_w}; //Prepare for use with computeCov2DCUDA_device
	float dr_dopacity = J[index*10 + 5];
    float3 dr_dmean2D = {dL_dmean2D_x, dL_dmean2D_y, 0.0f};

    float dL_dcolors[C] = {0.0f};
    for (int ch = 0; ch < C; ++ch) {
        dL_dcolors[ch] = J[index*10 + ch + 6];
    }
    float dL_dinvdepths = J[index*10 + 9];


    float3 dr_dmean = {0.0f, 0.0f, 0.0f}; 
    glm::vec4 dr_drot = glm::vec4(0.0f);
    glm::vec3 dr_dscale = glm::vec3(0.0f);
    // float dr_dopacity;  
    glm::vec3 dr_ddc = glm::vec3(0.0f);
    glm::vec3 dr_dsh[15] = {};  
    float dr_dcov3D[6] = {}; 

    computeCov2DCUDA_device(
        P,
        (float3*)means3D,
        radii,
        cov3Ds,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        viewmatrix,
        opacities,
        &dL_dconics[0],
        &dr_dopacity,
        &dL_dinvdepths,
        &dr_dmean,
        &dr_dcov3D[0],
        antialiasing,
        x
    );

    preprocessCUDA_device<C>(
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
        (glm::vec3*)&dr_dmean,
        &dL_dcolors[0],
        &dr_dcov3D[0],
        &dr_ddc[0],
        (float*)&dr_dsh[0],
        &dr_dscale,
        &dr_drot,
        &dr_dopacity,
        x
    );

    const int num_params_per_gauss = 59;
    float residual = loss_residuals[y];

    // if(x == 0 && dr_dmean.x != 0){
    //     // printf("dr_dmean.x not zero: %g  residual: %g \n", dr_dmean.x, residual);
    //     printf("dL_dcolors x: %g  y: %g  z: %g \n", dL_dcolors[0], dL_dcolors[1], dL_dcolors[2]);
    // }

    // if(x == 0 && dr_dmean.x != 0){
    //     printf(" res[%d]  dr_dmean.x: %g  residual: %g , dr_dsh[0].x: %g \n", y, dr_dmean.x, residual, dr_dsh[0].x);
    // }
    
    // if(x == 0){
    //     printf("residual: %g , dr_dsh[0].x: %g \n", residual, dr_dsh[0].x);
    // }

    // if(x == 0 && dL_dcolors[0] != 0){
    //     printf("dL_dcolors x: %g  y: %g  z: %g");
    // }

    glm::vec3 scale = {scales[x + 0], scales[x + 1], scales[x + 2]};
    dr_dscale.x = dr_dscale.x * scale.x;
    dr_dscale.y = dr_dscale.y * scale.y;
    dr_dscale.z = dr_dscale.z * scale.z;

    float real_opacity_val = log(opacities[x]/(1-opacities[x]));
    dr_dopacity = dr_dopacity * sigmoid_grad(real_opacity_val);

    float4 rot_original = {rotations[x+0],rotations[x+1],rotations[x+2],rotations[x+3]};

    float4 dr_drot_un = dnormvdv(rot_original, float4{dr_drot.x, dr_drot.y, dr_drot.z, dr_drot.w});
    dr_drot.x = dr_drot_un.x;
    dr_drot.y = dr_drot_un.y;
    dr_drot.z = dr_drot_un.z;
    dr_drot.w = dr_drot_un.w;


    atomicAdd(&b[x * num_params_per_gauss + 0], -dr_dmean.x * residual);
    atomicAdd(&b[x * num_params_per_gauss + 1], -dr_dmean.y * residual);
    atomicAdd(&b[x * num_params_per_gauss + 2], -dr_dmean.z * residual);

    atomicAdd(&b[x * num_params_per_gauss + 3], -dr_dscale.x * residual);
    atomicAdd(&b[x * num_params_per_gauss + 4], -dr_dscale.y * residual);
    atomicAdd(&b[x * num_params_per_gauss + 5], -dr_dscale.z * residual);

    atomicAdd(&b[x * num_params_per_gauss + 6], -dr_drot.x * residual);
    atomicAdd(&b[x * num_params_per_gauss + 7], -dr_drot.y * residual);
    atomicAdd(&b[x * num_params_per_gauss + 8], -dr_drot.z * residual);
    atomicAdd(&b[x * num_params_per_gauss + 9], -dr_drot.w * residual);

    atomicAdd(&b[x * num_params_per_gauss + 10], -dr_dopacity * residual);

    atomicAdd(&b[x * num_params_per_gauss + 11], -dr_ddc.x * residual);
    atomicAdd(&b[x * num_params_per_gauss + 12], -dr_ddc.y * residual);
    atomicAdd(&b[x * num_params_per_gauss + 13], -dr_ddc.z * residual);

    for(int i = 0; i < max_coeffs; i++){
        atomicAdd(&b[x * num_params_per_gauss + 14 + i * 3 + 0], -dr_dsh[i].x * residual);
        atomicAdd(&b[x * num_params_per_gauss + 14 + i * 3 + 1], -dr_dsh[i].y * residual);
        atomicAdd(&b[x * num_params_per_gauss + 14 + i * 3 + 2], -dr_dsh[i].z * residual);
    }
    
    
}

template<uint32_t C>
__global__
void residual_dot_sum_temp(
    float* J, 
    float* v, 
    float* residual_dot_v, 
    uint32_t N, 
    uint32_t M,
    int P, 
    int D, 
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
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    const float* campos, //const glm::vec3* campos,
    bool antialiasing
){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= P*M) return;

    int x = idx % P;  // Gaussian index
    int y = idx / P;  // Pixel index

    // Calculate the residuals wrt parameters
    int index= x + y * P;
    float dL_dmean2D_x = J[index*10];
	float dL_dmean2D_y = J[index*10 + 1];
	float dL_dconic2D_x = J[index*10 + 2] ;
	float dconic2D_y = J[index*10 + 3];
	float dL_dconic2D_w = J[index*10 + 4];
    float dL_dconics[4] = {dL_dconic2D_x, dconic2D_y, 0.0f, dL_dconic2D_w}; //Prepare for use with computeCov2DCUDA_device
	float dr_dopacity = J[index*10 + 5];
    float3 dr_dmean2D = {dL_dmean2D_x, dL_dmean2D_y, 0.0f};

    float dL_dcolors[C] = {0.0f};
    for (int ch = 0; ch < C; ++ch) {
        dL_dcolors[ch] = J[index*10 + ch + 6];
    }
    float dL_dinvdepths = J[index*10 + 9];

    float3 dr_dmean = {0.0f, 0.0f, 0.0f}; 
    glm::vec4 dr_drot = glm::vec4(0.0f);
    glm::vec3 dr_dscale = glm::vec3(0.0f);
    glm::vec3 dr_ddc = glm::vec3(0.0f);
    glm::vec3 dr_dsh[15] = {};  
    float dr_dcov3D[6] = {}; 

    computeCov2DCUDA_device(
        P,
        (float3*)means3D,
        radii,
        cov3Ds,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        viewmatrix,
        opacities,
        &dL_dconics[0],
        &dr_dopacity,
        &dL_dinvdepths,
        &dr_dmean,
        &dr_dcov3D[0],
        antialiasing,
        x
    );

    preprocessCUDA_device<C>(
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
        (glm::vec3*)&dr_dmean,
        &dL_dcolors[0],
        &dr_dcov3D[0],
        &dr_ddc[0],
        (float*)&dr_dsh[0],
        &dr_dscale,
        &dr_drot,
        &dr_dopacity,
        x
    );


    const int num_params_per_gauss = 59;

    glm::vec3 scale = {scales[x + 0], scales[x + 1], scales[x + 2]};
    dr_dscale.x = dr_dscale.x * scale.x;
    dr_dscale.y = dr_dscale.y * scale.y;
    dr_dscale.z = dr_dscale.z * scale.z;

    float real_opacity_val = log(opacities[x]/(1-opacities[x]));
    dr_dopacity = dr_dopacity * sigmoid_grad(real_opacity_val);

    float4 rot_original = {rotations[x+0],rotations[x+1],rotations[x+2],rotations[x+3]};

    float4 dr_drot_un = dnormvdv(rot_original, float4{dr_drot.x, dr_drot.y, dr_drot.z, dr_drot.w});
    dr_drot.x = dr_drot_un.x;
    dr_drot.y = dr_drot_un.y;
    dr_drot.z = dr_drot_un.z;
    dr_drot.w = dr_drot_un.w;


    atomicAdd(&residual_dot_v[y], dr_dmean.x * v[x * num_params_per_gauss + 0]);
    atomicAdd(&residual_dot_v[y], dr_dmean.y * v[x * num_params_per_gauss + 1]);
    atomicAdd(&residual_dot_v[y], dr_dmean.z * v[x * num_params_per_gauss + 2]);

    atomicAdd(&residual_dot_v[y], dr_dscale.x * v[x * num_params_per_gauss + 3]);
    atomicAdd(&residual_dot_v[y], dr_dscale.y * v[x * num_params_per_gauss + 4]);
    atomicAdd(&residual_dot_v[y], dr_dscale.z * v[x * num_params_per_gauss + 5]);

    atomicAdd(&residual_dot_v[y], dr_drot.x * v[x * num_params_per_gauss + 6]);
    atomicAdd(&residual_dot_v[y], dr_drot.y * v[x * num_params_per_gauss + 7]);
    atomicAdd(&residual_dot_v[y], dr_drot.z * v[x * num_params_per_gauss + 8]);
    atomicAdd(&residual_dot_v[y], dr_drot.w * v[x * num_params_per_gauss + 9]);

    atomicAdd(&residual_dot_v[y], dr_dopacity * v[x * num_params_per_gauss + 10]);

    atomicAdd(&residual_dot_v[y], dr_ddc.x * v[x * num_params_per_gauss + 11]);
    atomicAdd(&residual_dot_v[y], dr_ddc.y * v[x * num_params_per_gauss + 12]);
    atomicAdd(&residual_dot_v[y], dr_ddc.z * v[x * num_params_per_gauss + 13]);

    for(int i = 0; i < max_coeffs; i++){
        atomicAdd(&residual_dot_v[y], dr_dsh[i].x * v[x * num_params_per_gauss + 14 + i * 3 + 0]);
        atomicAdd(&residual_dot_v[y], dr_dsh[i].y * v[x * num_params_per_gauss + 14 + i * 3 + 1]);
        atomicAdd(&residual_dot_v[y], dr_dsh[i].z * v[x * num_params_per_gauss + 14 + i * 3 + 2]);
    }


}

template<uint32_t C>
__global__
void sum_residuals_temp(
    float* Av, 
    float* residual_dot_v, 
    float* J, 
    uint32_t N, 
    uint32_t M,
    int P, 
    int D, 
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
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    const float* campos, //const glm::vec3* campos,
    bool antialiasing
){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= P*M) return;

    int x = idx % P;  // Gaussian index
    int y = idx / P;  // Pixel index

    // Calculate the residuals wrt parameters
    int index= x + y * P;
    float dL_dmean2D_x = J[index*10];
	float dL_dmean2D_y = J[index*10 + 1];
	float dL_dconic2D_x = J[index*10 + 2] ;
	float dconic2D_y = J[index*10 + 3];
	float dL_dconic2D_w = J[index*10 + 4];
    float dL_dconics[4] = {dL_dconic2D_x, dconic2D_y, 0.0f, dL_dconic2D_w}; //Prepare for use with computeCov2DCUDA_device
	float dr_dopacity = J[index*10 + 5];
    float3 dr_dmean2D = {dL_dmean2D_x, dL_dmean2D_y, 0.0f};

    float dL_dcolors[C] = {0.0f};
    for (int ch = 0; ch < C; ++ch) {
        dL_dcolors[ch] = J[index*10 + ch + 6];
    }
    float dL_dinvdepths = J[index*10 + 9];

    float3 dr_dmean = {0.0f, 0.0f, 0.0f}; 
    glm::vec4 dr_drot = glm::vec4(0.0f);
    glm::vec3 dr_dscale = glm::vec3(0.0f);
    glm::vec3 dr_ddc = glm::vec3(0.0f);
    glm::vec3 dr_dsh[15] = {};  
    float dr_dcov3D[6] = {}; 

    // if(isnan(dr_dmean.x) && y == 0){
    //     printf("DR_DMEAN3D: %g , DR_DMEAN2D: %g,  [%d]\n", dr_dmean.x, dr_dmean2D.x, index);
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
        &dL_dconics[0],
        &dr_dopacity,
        &dL_dinvdepths,
        &dr_dmean,
        &dr_dcov3D[0],
        antialiasing,
        x
    );

    // if(isnan(dr_dmean.x) && y == 0){
    //     printf("DR_DMEAN3D: %g , DR_DMEAN2D: %g, radii: %d,  [%d]\n", dr_dmean.x, dr_dmean2D.x, radii[x], index);
    // }

    preprocessCUDA_device<C>(
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
        (glm::vec3*)&dr_dmean,
        &dL_dcolors[0],
        &dr_dcov3D[0],
        &dr_ddc[0],
        (float*)&dr_dsh[0],
        &dr_dscale,
        &dr_drot,
        &dr_dopacity,
        x
    );

    const int num_params_per_gauss = 59;

    // if(isnan(dr_dmean.x) && y == 0){
    //     printf("DR_DMEAN3D: %g , DR_DMEAN2D: %g,  [%d]\n", dr_dmean.x, dr_dmean2D.x, index);
    // }

    // atomicAdd(&Av[x * num_params_per_gauss + 0], dr_dmean.x * residual_dot_v[y]);

    glm::vec3 scale = {scales[x + 0], scales[x + 1], scales[x + 2]};
    dr_dscale.x = dr_dscale.x * scale.x;
    dr_dscale.y = dr_dscale.y * scale.y;
    dr_dscale.z = dr_dscale.z * scale.z;

    float real_opacity_val = log(opacities[x]/(1-opacities[x]));
    dr_dopacity = dr_dopacity * sigmoid_grad(real_opacity_val);

    float4 rot_original = {rotations[x+0],rotations[x+1],rotations[x+2],rotations[x+3]};

    float4 dr_drot_un = dnormvdv(rot_original, float4{dr_drot.x, dr_drot.y, dr_drot.z, dr_drot.w});
    dr_drot.x = dr_drot_un.x;
    dr_drot.y = dr_drot_un.y;
    dr_drot.z = dr_drot_un.z;
    dr_drot.w = dr_drot_un.w;


    
    // dr_drot.x = dr_drot.x * expf(rot.x);
    // dr_drot.y = dr_drot.y * expf(rot.y);
    // dr_drot.z = dr_drot.z * expf(rot.z);
    // dr_drot.w = dr_drot.w * expf(rot.w);


    atomicAdd(&Av[x * num_params_per_gauss + 0], dr_dmean.x * residual_dot_v[y]);
    atomicAdd(&Av[x * num_params_per_gauss + 1], dr_dmean.y * residual_dot_v[y]);
    atomicAdd(&Av[x * num_params_per_gauss + 2], dr_dmean.z * residual_dot_v[y]);

    atomicAdd(&Av[x * num_params_per_gauss + 3], dr_dscale.x * residual_dot_v[y]);
    atomicAdd(&Av[x * num_params_per_gauss + 4], dr_dscale.y * residual_dot_v[y]);
    atomicAdd(&Av[x * num_params_per_gauss + 5], dr_dscale.z * residual_dot_v[y]);

    atomicAdd(&Av[x * num_params_per_gauss + 6], dr_drot.x * residual_dot_v[y]);
    atomicAdd(&Av[x * num_params_per_gauss + 7], dr_drot.y * residual_dot_v[y]);
    atomicAdd(&Av[x * num_params_per_gauss + 8], dr_drot.z * residual_dot_v[y]);
    atomicAdd(&Av[x * num_params_per_gauss + 9], dr_drot.w * residual_dot_v[y]);

    atomicAdd(&Av[x * num_params_per_gauss + 10], dr_dopacity * residual_dot_v[y]);

    atomicAdd(&Av[x * num_params_per_gauss + 11], dr_ddc.x * residual_dot_v[y]);
    atomicAdd(&Av[x * num_params_per_gauss + 12], dr_ddc.y * residual_dot_v[y]);
    atomicAdd(&Av[x * num_params_per_gauss + 13], dr_ddc.z * residual_dot_v[y]);

    for(int i = 0; i < max_coeffs; i++){
        atomicAdd(&Av[x * num_params_per_gauss + 14 + i * 3 + 0], dr_dsh[i].x * residual_dot_v[y]);
        atomicAdd(&Av[x * num_params_per_gauss + 14 + i * 3 + 1], dr_dsh[i].y * residual_dot_v[y]);
        atomicAdd(&Av[x * num_params_per_gauss + 14 + i * 3 + 2], dr_dsh[i].z * residual_dot_v[y]);
    }
    
}


void Av_temp(
    float* J, 
    float* v, 
    float* Av, 
    uint32_t N, 
    uint32_t M,
    int P, 
    int D, 
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
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    const float* campos, //const glm::vec3* campos,
    bool antialiasing
){
    float* residual_dot_v;
    cudaMalloc(&residual_dot_v, M * sizeof(float));
    cudaMemset(residual_dot_v, 0, M * sizeof(float));
    
    residual_dot_sum_temp<NUM_CHANNELS_3DGS><<<((P*M)+255)/256, 256>>>(
        J, 
        v, 
        residual_dot_v, 
        N, 
        M,
        P, 
        D, 
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
        focal_x, 
        focal_y, 
        tan_fovx, 
        tan_fovy, 
        campos, 
        antialiasing
    );    // (r(i)^T * p)


    sum_residuals_temp<NUM_CHANNELS_3DGS><<<((P*M)+255)/256, 256>>>(
        Av, 
        residual_dot_v, 
        J, 
        N, 
        M,
        P, 
        D, 
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
        focal_x, 
        focal_y, 
        tan_fovx, 
        tan_fovy, 
        campos, 
        antialiasing
    );    // Ap = r(i) * (r(i)^T * p)
    cudaFree(residual_dot_v);
}


template<uint32_t C>
__global__ 
void diagJTJ_temp(
    const float* J, 
    float* M_precon, 
    uint32_t N, 
    uint32_t M,
    int P, 
    int D, 
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
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    const float* campos, //const glm::vec3* campos,
    bool antialiasing
){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= P*M) return;
    int x = idx % P;  // Gaussian index
    int y = idx / P;  // Pixel index


    // Calculate the residuals wrt parameters
    int index= x + y * P;
    float dL_dmean2D_x = J[index*10];
	float dL_dmean2D_y = J[index*10 + 1];
	float dL_dconic2D_x = J[index*10 + 2] ;
	float dconic2D_y = J[index*10 + 3];
	float dL_dconic2D_w = J[index*10 + 4];
    float dL_dconics[4] = {dL_dconic2D_x, dconic2D_y, 0.0f, dL_dconic2D_w}; //Prepare for use with computeCov2DCUDA_device
	float dr_dopacity = J[index*10 + 5];
    float3 dr_dmean2D = {dL_dmean2D_x, dL_dmean2D_y, 0.0f};

    float dL_dcolors[C] = {0.0f};
    for (int ch = 0; ch < C; ++ch) {
        dL_dcolors[ch] = J[index*10 + ch + 6];
    }
    float dL_dinvdepths = J[index*10 + 9];



    float3 dr_dmean = {0.0f, 0.0f, 0.0f}; 
    glm::vec4 dr_drot = glm::vec4(0.0f);
    glm::vec3 dr_dscale = glm::vec3(0.0f);
    glm::vec3 dr_ddc = glm::vec3(0.0f);
    glm::vec3 dr_dsh[15] = {};  
    float dr_dcov3D[6] = {}; 


    computeCov2DCUDA_device(
        P,
        (float3*)means3D,
        radii,
        cov3Ds,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        viewmatrix,
        opacities,
        &dL_dconics[0],
        &dr_dopacity,
        &dL_dinvdepths,
        &dr_dmean,
        &dr_dcov3D[0],
        antialiasing,
        x
    );

    preprocessCUDA_device<C>(
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
        (glm::vec3*)&dr_dmean,
        &dL_dcolors[0],
        &dr_dcov3D[0],
        &dr_ddc[0],
        (float*)&dr_dsh[0],
        &dr_dscale,
        &dr_drot,
        &dr_dopacity,
        x
    );



    // float j_val = J[x + y * N];

    // float tmp = j_val * j_val;

    // atomicAdd(&M_precon[x],tmp);

    // dr_dmean = grad_sigmoid(dr_dmean)

    const int num_params_per_gauss = 59;

    glm::vec3 scale = {scales[x + 0], scales[x + 1], scales[x + 2]};
    dr_dscale.x = dr_dscale.x * scale.x;
    dr_dscale.y = dr_dscale.y * scale.y;
    dr_dscale.z = dr_dscale.z * scale.z;

    float real_opacity_val = log(opacities[x]/(1-opacities[x]));
    dr_dopacity = dr_dopacity * sigmoid_grad(real_opacity_val);

    float4 rot_original = {rotations[x+0],rotations[x+1],rotations[x+2],rotations[x+3]};

    float4 dr_drot_un = dnormvdv(rot_original, float4{dr_drot.x, dr_drot.y, dr_drot.z, dr_drot.w});
    dr_drot.x = dr_drot_un.x;
    dr_drot.y = dr_drot_un.y;
    dr_drot.z = dr_drot_un.z;
    dr_drot.w = dr_drot_un.w;

    // atomicAdd(&M_precon[x + 0], dr_dmean.x * dr_dmean.x);
    // atomicAdd(&M_precon[x + 1], dr_dmean.y * dr_dmean.y);
    // atomicAdd(&M_precon[x + 2], dr_dmean.z * dr_dmean.z);

    // atomicAdd(&M_precon[x + 3], dr_dscale.x * dr_dscale.x);
    // atomicAdd(&M_precon[x + 4], dr_dscale.y * dr_dscale.y);
    // atomicAdd(&M_precon[x + 5], dr_dscale.z * dr_dscale.z);

    // atomicAdd(&M_precon[x + 6], dr_drot.x * dr_drot.x);
    // atomicAdd(&M_precon[x + 7], dr_drot.y * dr_drot.y);
    // atomicAdd(&M_precon[x + 8], dr_drot.z * dr_drot.z);
    // atomicAdd(&M_precon[x + 9], dr_drot.w * dr_drot.w);

    // atomicAdd(&M_precon[x + 10], dr_dopacity * dr_dopacity);

    // atomicAdd(&M_precon[x + 11], dr_ddc.x * dr_ddc.x);
    // atomicAdd(&M_precon[x + 12], dr_ddc.y * dr_ddc.y);
    // atomicAdd(&M_precon[x + 13], dr_ddc.z * dr_ddc.z);

    // for(int i = 0; i < max_coeffs; i++){
    //     atomicAdd(&M_precon[y], dr_dsh[i].x * dr_dsh[i].x);
    //     atomicAdd(&M_precon[y], dr_dsh[i].y * dr_dsh[i].y);
    //     atomicAdd(&M_precon[y], dr_dsh[i].z * dr_dsh[i].z);
    // }

    atomicAdd(&M_precon[y], dr_dmean.x * dr_dmean.x);
    atomicAdd(&M_precon[y], dr_dmean.y * dr_dmean.y);
    atomicAdd(&M_precon[y], dr_dmean.z * dr_dmean.z);

    atomicAdd(&M_precon[y], dr_dscale.x * dr_dscale.x);
    atomicAdd(&M_precon[y], dr_dscale.y * dr_dscale.y);
    atomicAdd(&M_precon[y], dr_dscale.z * dr_dscale.z);

    atomicAdd(&M_precon[y], dr_drot.x * dr_drot.x);
    atomicAdd(&M_precon[y], dr_drot.y * dr_drot.y);
    atomicAdd(&M_precon[y], dr_drot.z * dr_drot.z);
    atomicAdd(&M_precon[y], dr_drot.w * dr_drot.w);

    atomicAdd(&M_precon[y], dr_dopacity * dr_dopacity);

    atomicAdd(&M_precon[y], dr_ddc.x * dr_ddc.x);
    atomicAdd(&M_precon[y], dr_ddc.y * dr_ddc.y);
    atomicAdd(&M_precon[y], dr_ddc.z * dr_ddc.z);

    for(int i = 0; i < max_coeffs; i++){
        atomicAdd(&M_precon[y], dr_dsh[i].x * dr_dsh[i].x);
        atomicAdd(&M_precon[y], dr_dsh[i].y * dr_dsh[i].y);
        atomicAdd(&M_precon[y], dr_dsh[i].z * dr_dsh[i].z);
    }
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
	const float tan_fovx, float tan_fovy,
    const float* campos, //const glm::vec3* campos,
    bool antialiasing,

    

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
    ){

    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);
    // TODO: create b from sparse
    // TODO: create M / M^-1
    // TODO: create Ap, following perf or levenberg
    // TODO: allow multiple images (stack sparse jacobians)
    // TODO: apply activation functions?


    //Calculate residuals wrt parameters

    float h_float;

     

    
    // b = JTv (v = r) = JTr, b=-JTr
    float* J = sparse_J_values;
    // for(int i = 0; i < 11*59; i++){
    //     cudaMemcpy(&h_float, &J[i], sizeof(float), cudaMemcpyDeviceToHost);
    //     printf("J(%d): %g \n",i, h_float);
    // }


    float* b;
    cudaMalloc(&b, N * sizeof(float));
    cudaMemset(b,0, N * sizeof(float));

    calc_b_non_sparse<NUM_CHANNELS_3DGS><<<(P*M+255)/256, 256>>>(
        b,
        J, 
        loss_residuals, 
        N, 
        M, 
        P, 
        D, 
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
        focal_x, 
        focal_y, 
        tan_fovx, 
        tan_fovy, 
        campos, 
        antialiasing
    );

    // printf("Max coeffs: %d", max_coeffs);

    // for(int i = 0; i < 11*59; i++){
    //     cudaMemcpy(&h_float, &b[i], sizeof(float), cudaMemcpyDeviceToHost);
    //     printf("b(%d): %g  | %s \n",i, h_float, get_param_name(i%59));
    // }
    
    // return;


    // return;
    // r(0) = b
    float* dev_r;
    cudaMalloc(&dev_r, N * sizeof(float));
    gpu_copy<<<(N+255)/256, 256>>>(b, dev_r, N);

    // calculate A*x(0)
    float* dev_Ax0;
    cudaMalloc(&dev_Ax0, N * sizeof(float));
    cudaMemset(dev_Ax0, 0, N * sizeof(float));

    // for(int i = 0; i < 100; i++){
    //     cudaMemcpy(&h_float, &dev_Ax0[i], sizeof(float), cudaMemcpyDeviceToHost);
    //     printf("Ax0_prev(%d): %g \n",i, h_float);
    // }
    // Av(J, x, dev_Ax0, N, M); //Ax
    // Av_temp(
    //     J, 
    //     x, 
    //     dev_Ax0, 
    //     N, 
    //     M,
    //     P, 
    //     D, 
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
    //     focal_x, 
    //     focal_y, 
    //     tan_fovx, 
    //     tan_fovy, 
    //     campos, 
    //     antialiasing
    // ); //Ax

    // for(int i = 0; i < 100; i++){
    //     cudaMemcpy(&h_float, &dev_Ax0[i], sizeof(float), cudaMemcpyDeviceToHost);
    //     printf("Ax0(%d): %g \n",i, h_float);
    // }

    // r(0) = b - A*x(0)
    subtract<<<(N+255)/256, 256>>>(dev_r, dev_Ax0, dev_r, N);

    
    // calculate M = diag(J^T * J)
    float* M_precon;
    cudaMalloc(&M_precon, N * sizeof(float));
    cudaMemset(M_precon, 0, N * sizeof(float));
    dim3 threadsPerBlock(256); // thread block size: 256
    dim3 numBlocks(((N*M)+255)/256); 
    // diagJTJ<<<numBlocks, threadsPerBlock>>>(J, M_precon, N, M);
    diagJTJ_temp<NUM_CHANNELS_3DGS><<<((P*M)+255)/256, 256>>>(
        J, 
        M_precon, 
        N, 
        M,
        P, 
        D, 
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
        focal_x, 
        focal_y, 
        tan_fovx, 
        tan_fovy, 
        campos, 
        antialiasing
    );

    // for(int i = 0; i < 100; i++){
    //     cudaMemcpy(&h_float, &dev_r[i], sizeof(float), cudaMemcpyDeviceToHost);
    //     printf("r(%d): %g \n",i, h_float);
    // }

    // for(int i = 0; i < 100; i++){
    //     cudaMemcpy(&h_float, &M_precon[i], sizeof(float), cudaMemcpyDeviceToHost);
    //     printf("M_precon(%d): %g \n",i, h_float);
    // }
    
    // z(0) = M^-1 * r(0)
    float* dev_z;
    cudaMalloc(&dev_z, N * sizeof(float));
    next_z<<<(N+255)/256, 256>>>(M_precon, dev_r, dev_z, N);

    // float h_float;
    // for(int i = 0; i < 100; i++){
    //     cudaMemcpy(&h_float, &dev_z[i], sizeof(float), cudaMemcpyDeviceToHost);
    //     printf("z(%d): %g \n",i, h_float);
    // }

    // p(0) = z(0)
    float* dev_p;
    cudaMalloc(&dev_p, N * sizeof(float));
    gpu_copy<<<(N+255)/256, 256>>>(dev_z, dev_p, N); //#change back

    // for(int i = 0; i < 100; i++){
    //     cudaMemcpy(&h_float, &dev_p[i], sizeof(float), cudaMemcpyDeviceToHost);
    //     printf("p(%d): %g \n",i, h_float);
    // }
    
    
    float eps = 0.00000001;
    float* h_R;
    float* h_R_prev;
    float* dev_R;
    float* dev_R_prev;
    
    cudaMalloc(&dev_R, sizeof(float));
    cudaMalloc(&dev_R_prev, sizeof(float));

    float* dev_Ap;
    cudaMalloc(&dev_Ap, N * sizeof(float));
    
    h_R = (float *)malloc(sizeof(float));
    h_R_prev = (float *)malloc(sizeof(float));

    cudaMemset(dev_R, 0, sizeof(float));
    cudaMemset(dev_R_prev, 0, sizeof(float));
    

    // R_prev = r(0)^T * r(0)
    dot<<<(N+255)/256, 256>>>(dev_r, dev_r, dev_R_prev, N);
    
    float* dev_alpha;
    double* dev_denominator;
    double* dev_numerator;
    float* dev_beta;
    cudaMalloc(&dev_alpha, sizeof(float));
    cudaMalloc(&dev_denominator, sizeof(double));
    cudaMalloc(&dev_numerator, sizeof(double));
    cudaMalloc(&dev_beta, sizeof(float));

    
    int k = 0; 
    const int MAX_ITERATIONS = 5;
    cudaMemcpy(h_R_prev, dev_R_prev, sizeof(float), cudaMemcpyDeviceToHost);

    
    while(k < MAX_ITERATIONS){

        cudaMemset(dev_alpha, 0, sizeof(float));
        cudaMemset(dev_denominator, 0, sizeof(float));
        cudaMemset(dev_numerator, 0, sizeof(double));
        cudaMemset(dev_beta, 0, sizeof(float));
        cudaMemset(dev_Ap, 0, N * sizeof(float));

        // float h_float;
        // for(int i = 0; i < 100; i++){
        //     cudaMemcpy(&h_float, &dev_r[i], sizeof(float), cudaMemcpyDeviceToHost);
        //     printf("r(%d): %g \n",i, h_float);
        // }   

        // float h_float;
        // for(int i = 0; i < 100; i++){
        //     cudaMemcpy(&h_float, &dev_z[i], sizeof(float), cudaMemcpyDeviceToHost);
        //     printf("z(%d): %g \n",i, h_float);
        // }   

        // r(k)^T * z(k)
        dot_d<<<(N+255)/256, 256>>>(dev_r, dev_z, dev_numerator, N);

        // float h_float;
        double h_double;
        // cudaMemcpy(&h_double, dev_numerator, sizeof(double), cudaMemcpyDeviceToHost);
        // printf("numerator(%d): %g \n",0, h_double);

        // for(int i = 0; i < 100; i++){
        //     cudaMemcpy(&h_float, &J[i], sizeof(float), cudaMemcpyDeviceToHost);
        //     printf("J(%d): %g \n",i, h_float);
        // }
        // A*p(k)
        // Av(J, dev_p, dev_Ap, N, M);
        Av_temp(
            J, 
            dev_p, 
            dev_Ap, 
            N, 
            M,
            P, 
            D, 
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
            focal_x, 
            focal_y, 
            tan_fovx, 
            tan_fovy, 
            campos, 
            antialiasing
        );

        // float h_float;
        // for(int i = 0; i < 100; i++){
        //     cudaMemcpy(&h_float, &dev_Ap[i], sizeof(float), cudaMemcpyDeviceToHost);
        //     printf("Ap(%d): %g \n",i, h_float);
        // }

        // for(int i = 0; i < 100; i++){
        //     cudaMemcpy(&h_float, &dev_p[i], sizeof(float), cudaMemcpyDeviceToHost);
        //     printf("p(%d): %g \n",i, h_float);
        // }

        
        // p(k)^T * Ap(k) 
        dot_d<<<(N+255)/256, 256>>>(dev_p, dev_Ap, dev_denominator, N);

        // float h_float;
        // cudaMemcpy(&h_double, dev_numerator, sizeof(double), cudaMemcpyDeviceToHost);
        // printf("numerator(%d): %g \n",0, h_double);

        
        // cudaMemcpy(&h_double, dev_denominator, sizeof(double), cudaMemcpyDeviceToHost);
        // printf("denomenator(%d): %g \n",0, h_double);
       
        // alpha = r(k)^T * z(k) / p(k)^T * Ap(k) 
        scalar_divide_double<<<1, 1>>>(dev_numerator, dev_denominator, dev_alpha); 

        // float h_float_a;
        // cudaMemcpy(&h_float_a, dev_alpha, sizeof(float), cudaMemcpyDeviceToHost);
        // printf("alpha(%d): %g \n",0, h_float_a);
        
        // x(k+1) = x + alpha * p
        next_x<<<(N+255)/256, 256>>>(x, dev_p, dev_alpha, N);

        // float h_float;
        // for(int i = 0; i < 100; i++){
        //     cudaMemcpy(&h_float, &x[i], sizeof(float), cudaMemcpyDeviceToHost);
        //     printf("x(%d): %g \n",i, h_float);
        // }


        // r(k)^T * z(k)  (used for beta calculation)
        cudaMemset(dev_denominator, 0, sizeof(float));
        dot_d<<<(N+255)/256, 256>>>(dev_r, dev_z, dev_denominator, N); 

        // r(k+1) = r(k) - alpha * Ap(k)
        next_r<<<(N+255)/256, 256>>>(dev_r, dev_Ap, dev_alpha, N);

        // R = r(k+1)^T * r(k+1)
        dot<<<(N+255)/256, 256>>>(dev_r, dev_r, dev_R, N); 
        
        // Check if R/Rprev > 0.85 or R < eps
        cudaMemcpy(h_R, dev_R, sizeof(float), cudaMemcpyDeviceToHost);
        // printf("R: %g, Rprev: %g \n", *h_R, *h_R_prev);
        printf("R/R_prev: %g \n", (*h_R/ *h_R_prev));
        // if (*h_R/ *h_R_prev > 0.85f){
        //     break;
        // }
        
        *h_R_prev = *h_R;

        // z(k+1) = M^-1 * r(k)
        next_z<<<(N+255)/256, 256>>>(M_precon, dev_r, dev_z, N);
        
        // r(k+1)^T * z(k+1)
        cudaMemset(dev_numerator, 0, sizeof(float));
        dot_d<<<(N+255)/256, 256>>>(dev_r, dev_z, dev_numerator, N); 
        
        // beta = r(k+1)^T * z(k+1) / r(k)^T * z(k)
        scalar_divide_double<<<1, 1>>>(dev_numerator, dev_denominator, dev_beta); 

        // p(k+1) = z(k) + beta * p(k)
        next_p<<<(N+255)/256, 256>>>(dev_z, dev_p, dev_beta, N);


        k++;

    }

    //Free memory
    free(h_R);
    free(h_R_prev);
    // free(h_float);

    cudaFree(dev_p);
    cudaFree(dev_R);
    cudaFree(dev_R_prev);
    cudaFree(dev_Ap);
    cudaFree(dev_Ax0);
    cudaFree(dev_alpha);
    cudaFree(dev_denominator);
    cudaFree(dev_numerator);
    cudaFree(dev_beta);
    cudaFree(dev_z);
    cudaFree(M_precon);
    cudaFree(dev_r);
    cudaFree(b);


}


    // printing variable
    // float* h_float;
    // h_float = (float *)malloc(sizeof(float));
    // cudaMemcpy(h_float, &x[0], sizeof(float), cudaMemcpyDeviceToHost);
    // printf("x(0): %g \n", *h_float);
