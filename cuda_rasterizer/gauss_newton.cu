#include "auxiliary.h"
#include "gauss_newton.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 256

// // Backward pass for conversion of spherical harmonics to RGB for
// // each Gaussian.
// __device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* dc, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_ddc, glm::vec3* dL_dshs)
// {
// 	// Compute intermediate values, as it is done during forward
// 	glm::vec3 pos = means[idx];
// 	glm::vec3 dir_orig = pos - campos;
// 	glm::vec3 dir = dir_orig / glm::length(dir_orig);

// 	glm::vec3* direct_color = ((glm::vec3*)dc) + idx;
// 	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

// 	// Use PyTorch rule for clamping: if clamping was applied,
// 	// gradient becomes 0.
// 	glm::vec3 dL_dRGB = dL_dcolor[idx];
// 	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
// 	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
// 	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

// 	glm::vec3 dRGBdx(0, 0, 0);
// 	glm::vec3 dRGBdy(0, 0, 0);
// 	glm::vec3 dRGBdz(0, 0, 0);
// 	float x = dir.x;
// 	float y = dir.y;
// 	float z = dir.z;

// 	// Target location for this Gaussian to write SH gradients to
// 	glm::vec3* dL_ddirect_color = dL_ddc + idx;
// 	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

// 	// No tricks here, just high school-level calculus.
// 	float dRGBdsh0 = SH_C0;
// 	dL_ddirect_color[0] = dRGBdsh0 * dL_dRGB;
// 	if (deg > 0)
// 	{
// 		float dRGBdsh1 = -SH_C1 * y;
// 		float dRGBdsh2 = SH_C1 * z;
// 		float dRGBdsh3 = -SH_C1 * x;
// 		dL_dsh[0] = dRGBdsh1 * dL_dRGB;
// 		dL_dsh[1] = dRGBdsh2 * dL_dRGB;
// 		dL_dsh[2] = dRGBdsh3 * dL_dRGB;

// 		dRGBdx = -SH_C1 * sh[2];
// 		dRGBdy = -SH_C1 * sh[0];
// 		dRGBdz = SH_C1 * sh[1];

// 		if (deg > 1)
// 		{
// 			float xx = x * x, yy = y * y, zz = z * z;
// 			float xy = x * y, yz = y * z, xz = x * z;

// 			float dRGBdsh4 = SH_C2[0] * xy;
// 			float dRGBdsh5 = SH_C2[1] * yz;
// 			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
// 			float dRGBdsh7 = SH_C2[3] * xz;
// 			float dRGBdsh8 = SH_C2[4] * (xx - yy);
// 			dL_dsh[3] = dRGBdsh4 * dL_dRGB;
// 			dL_dsh[4] = dRGBdsh5 * dL_dRGB;
// 			dL_dsh[5] = dRGBdsh6 * dL_dRGB;
// 			dL_dsh[6] = dRGBdsh7 * dL_dRGB;
// 			dL_dsh[7] = dRGBdsh8 * dL_dRGB;

// 			dRGBdx += SH_C2[0] * y * sh[3] + SH_C2[2] * 2.f * -x * sh[5] + SH_C2[3] * z * sh[6] + SH_C2[4] * 2.f * x * sh[7];
// 			dRGBdy += SH_C2[0] * x * sh[3] + SH_C2[1] * z * sh[4] + SH_C2[2] * 2.f * -y * sh[5] + SH_C2[4] * 2.f * -y * sh[7];
// 			dRGBdz += SH_C2[1] * y * sh[4] + SH_C2[2] * 2.f * 2.f * z * sh[5] + SH_C2[3] * x * sh[6];

// 			if (deg > 2)
// 			{
// 				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
// 				float dRGBdsh10 = SH_C3[1] * xy * z;
// 				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
// 				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
// 				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
// 				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
// 				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
// 				dL_dsh[8] = dRGBdsh9 * dL_dRGB;
// 				dL_dsh[9] = dRGBdsh10 * dL_dRGB;
// 				dL_dsh[10] = dRGBdsh11 * dL_dRGB;
// 				dL_dsh[11] = dRGBdsh12 * dL_dRGB;
// 				dL_dsh[12] = dRGBdsh13 * dL_dRGB;
// 				dL_dsh[13] = dRGBdsh14 * dL_dRGB;
// 				dL_dsh[14] = dRGBdsh15 * dL_dRGB;

// 				dRGBdx += (
// 					SH_C3[0] * sh[8] * 3.f * 2.f * xy +
// 					SH_C3[1] * sh[9] * yz +
// 					SH_C3[2] * sh[10] * -2.f * xy +
// 					SH_C3[3] * sh[11] * -3.f * 2.f * xz +
// 					SH_C3[4] * sh[12] * (-3.f * xx + 4.f * zz - yy) +
// 					SH_C3[5] * sh[13] * 2.f * xz +
// 					SH_C3[6] * sh[14] * 3.f * (xx - yy));

// 				dRGBdy += (
// 					SH_C3[0] * sh[8] * 3.f * (xx - yy) +
// 					SH_C3[1] * sh[9] * xz +
// 					SH_C3[2] * sh[10] * (-3.f * yy + 4.f * zz - xx) +
// 					SH_C3[3] * sh[11] * -3.f * 2.f * yz +
// 					SH_C3[4] * sh[12] * -2.f * xy +
// 					SH_C3[5] * sh[13] * -2.f * yz +
// 					SH_C3[6] * sh[14] * -3.f * 2.f * xy);

// 				dRGBdz += (
// 					SH_C3[1] * sh[9] * xy +
// 					SH_C3[2] * sh[10] * 4.f * 2.f * yz +
// 					SH_C3[3] * sh[11] * 3.f * (2.f * zz - xx - yy) +
// 					SH_C3[4] * sh[12] * 4.f * 2.f * xz +
// 					SH_C3[5] * sh[13] * (xx - yy));
// 			}
// 		}
// 	}

// 	// The view direction is an input to the computation. View direction
// 	// is influenced by the Gaussian's mean, so SHs gradients
// 	// must propagate back into 3D position.
// 	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

// 	// Account for normalization of direction
// 	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

// 	// Gradients of loss w.r.t. Gaussian means, but only the portion 
// 	// that is caused because the mean affects the view-dependent color.
// 	// Additional mean gradient is accumulated in below methods.
// 	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
// }

// // Backward pass for the conversion of scale and rotation to a 
// // 3D covariance matrix for each Gaussian. 
// __device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
// {
// 	// Recompute (intermediate) results for the 3D covariance computation.
// 	glm::vec4 q = rot;// / glm::length(rot);
// 	float r = q.x;
// 	float x = q.y;
// 	float y = q.z;
// 	float z = q.w;

// 	glm::mat3 R = glm::mat3(
// 		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
// 		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
// 		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
// 	);

// 	glm::mat3 S = glm::mat3(1.0f);

// 	glm::vec3 s = mod * scale;
// 	S[0][0] = s.x;
// 	S[1][1] = s.y;
// 	S[2][2] = s.z;

// 	glm::mat3 M = S * R;

// 	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

// 	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
// 	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

// 	// Convert per-element covariance loss gradients to matrix form
// 	glm::mat3 dL_dSigma = glm::mat3(
// 		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
// 		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
// 		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
// 	);

// 	// Compute loss gradient w.r.t. matrix M
// 	// dSigma_dM = 2 * M
// 	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

// 	glm::mat3 Rt = glm::transpose(R);
// 	glm::mat3 dL_dMt = glm::transpose(dL_dM);

// 	// Gradients of loss w.r.t. scale
// 	glm::vec3* dL_dscale = dL_dscales + idx;
// 	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
// 	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
// 	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

// 	dL_dMt[0] *= s.x;
// 	dL_dMt[1] *= s.y;
// 	dL_dMt[2] *= s.z;

// 	// Gradients of loss w.r.t. normalized quaternion
// 	glm::vec4 dL_dq;
// 	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
// 	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
// 	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
// 	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

// 	// Gradients of loss w.r.t. unnormalized quaternion
// 	float4* dL_drot = (float4*)(dL_drots + idx);
// 	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
// }

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
    float inv = 1.0 / denominator;
    if (denominator == 0.0f){
        inv = 1.0;
    }
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

__global__
void calc_b_non_sparse(float* b, float* J, float* loss_residuals, uint32_t N, uint32_t M){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N*M) return;
    uint32_t x = idx % N;
    uint32_t y = idx / N;
    float dr_dG = J[x + y * N];
    glm::vec3 dr_dmean;


}

void GaussNewton::gaussNewtonUpdate(
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


    // TODO: create b from sparse
    // TODO: create M / M^-1
    // TODO: create Ap, following perf or levenberg
    // TODO: allow multiple images (stack sparse jacobians)

    //Calculate residuals wrt parameters
    
    // b = JTv (v = r) = JTr, b=-JTr


    float* J = sparse_J_values;
    float* b = x;
    // r(0) = b
    float* dev_r;
    cudaMalloc(&dev_r, N * sizeof(float));
    gpu_copy<<<(N+255)/256, 256>>>(b, dev_r, N);

    // calculate A*x(0)
    float* dev_Ax0;
    cudaMalloc(&dev_Ax0, N * sizeof(float));
    Av(J, x, dev_Ax0, N, M); //Ax

    // r(0) = b - A*x(0)
    subtract<<<(N+255)/256, 256>>>(dev_r, dev_Ax0, dev_r, N);

    
    // calculate M = diag(J^T * J)
    float* M_precon;
    cudaMalloc(&M_precon, N * sizeof(float));
    dim3 threadsPerBlock(256); // thread block size: 256
    dim3 numBlocks(((N*M)+255)/256); 
    diagJTJ<<<numBlocks, threadsPerBlock>>>(J, M_precon, N, M);
    
    // z(0) = M^-1 * r(0)
    float* dev_z;
    cudaMalloc(&dev_z, N * sizeof(float));
    next_z<<<(N+255)/256, 256>>>(M_precon, dev_r, dev_z, N);

    // p(0) = z(0)
    float* dev_p;
    cudaMalloc(&dev_p, N * sizeof(float));
    gpu_copy<<<(N+255)/256, 256>>>(dev_z, dev_p, N); //#change back
    
    
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
    float* dev_denominator;
    float* dev_numerator;
    float* dev_beta;
    cudaMalloc(&dev_alpha, sizeof(float));
    cudaMalloc(&dev_denominator, sizeof(float));
    cudaMalloc(&dev_numerator, sizeof(float));
    cudaMalloc(&dev_beta, sizeof(float));

    
    int k = 0; 
    const int MAX_ITERATIONS = 5;
    cudaMemcpy(h_R_prev, dev_R_prev, sizeof(float), cudaMemcpyDeviceToHost);

    
    while(k < MAX_ITERATIONS){

        cudaMemset(dev_alpha, 0, sizeof(float));
        cudaMemset(dev_denominator, 0, sizeof(float));
        cudaMemset(dev_numerator, 0, sizeof(float));
        cudaMemset(dev_beta, 0, sizeof(float));

        // r(k)^T * z(k)
        dot<<<(N+255)/256, 256>>>(dev_r, dev_z, dev_numerator, N);

        // A*p(k)
        Av(J, dev_p, dev_Ap, N, M); 

        // p(k)^T * Ap(k) 
        dot<<<(N+255)/256, 256>>>(dev_p, dev_Ap, dev_denominator, N);
       
        // alpha = r(k)^T * z(k) / p(k)^T * Ap(k) 
        scalar_divide<<<1, 1>>>(dev_numerator, dev_denominator, dev_alpha); 
        
        // x(k+1) = x + alpha * p
        next_x<<<(N+255)/256, 256>>>(x, dev_p, dev_alpha, N);


        // r(k)^T * z(k)  (used for beta calculation)
        cudaMemset(dev_denominator, 0, sizeof(float));
        dot<<<(N+255)/256, 256>>>(dev_r, dev_z, dev_denominator, N); 

        // r(k+1) = r(k) - alpha * Ap(k)
        next_r<<<(N+255)/256, 256>>>(dev_r, dev_Ap, dev_alpha, N);

        // R = r(k+1)^T * r(k+1)
        dot<<<(N+255)/256, 256>>>(dev_r, dev_r, dev_R, N); 
        
        // Check if R/Rprev > 0.85 or R < eps
        cudaMemcpy(h_R, dev_R, sizeof(float), cudaMemcpyDeviceToHost);
        // printf("R: %g, Rprev: %g \n", *h_R, *h_R_prev);
        // printf("R/R_prev: %g \n", (*h_R/ *h_R_prev));
        if (*h_R/ *h_R_prev > 0.85f){
            break;
        }
        
        *h_R_prev = *h_R;

        // z(k+1) = M^-1 * r(k)
        next_z<<<(N+255)/256, 256>>>(M_precon, dev_r, dev_z, N);
        
        // r(k+1)^T * z(k+1)
        cudaMemset(dev_numerator, 0, sizeof(float));
        dot<<<(N+255)/256, 256>>>(dev_r, dev_z, dev_numerator, N); 
        
        // beta = r(k+1)^T * z(k+1) / r(k)^T * z(k)
        scalar_divide<<<1, 1>>>(dev_numerator, dev_denominator, dev_beta); 

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


}


    // printing variable
    // float* h_float;
    // h_float = (float *)malloc(sizeof(float));
    // cudaMemcpy(h_float, &x[0], sizeof(float), cudaMemcpyDeviceToHost);
    // printf("x(0): %g \n", *h_float);
