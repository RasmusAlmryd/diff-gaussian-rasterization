#include "auxiliary.h"
#include "gauss_newton.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 256


__global__
void createJ(){

}

__global__ void diagJTJ(const float* J, float* M_precon, uint32_t N, uint32_t M){
    uint32_t idx = blockIdx.x * blockdim.x + threadIdx.x;
    if(idx >= N*M) return;
    uint32_t x = idx % N;
    uint32_t y = idx / N;


    float j_val = J[x + y * N];

    float tmp = j_val * j_val;

    atomicAdd(&M_precon[x],tmp);
}

__global__ void next_z(float* M_precon, float* r_0, float* z, uint32_t N){
    uint32_t idx = blockIdx.x * blockdim.x + threadIdx.x;
    if(idx >= N) return;
    float inv = 1.0 / M[idx];
    float z_val = inv * r_0[idx];
    z[idx] = z_val;
}

// __global__ void rTr(float* r, float* R, uint32_t N){
//     float r_val = r[N];
//     atomicAdd(R, r_val*r_val);
// }

__global__ void dot(int *a, int *b, int *c, uint32_t N){
    __shared__ int temp[THREADS_PER_BLOCK];
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= N) return;

    temp[threadIdx.x] = a[idx] * b[idx];

    __syncthreads();

    if (threadIdx.x == 0)
    {
        int sum = 0;
        for (int i = 0; i < N; i++)
        {
            sum += temp[i];
        }
        atomicAdd(c, sum);
    }
}

// template <unsigned int blockSize>
// __device__ void warpReduce(volatile int *sdata, unsigned int tid) {
//     if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//     if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//     if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//     if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
//     if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//     if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
// }


// template <unsigned int blockSize>
// __global__ void dot(int *g_idata, int *g_odata, unsigned int n) {
//     extern __shared__ int sdata[];
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x*(blockSize*2) + tid;
//     unsigned int gridSize = blockSize*2*gridDim.x;
//     sdata[tid] = 0;
//     while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
//     __syncthreads();
//     if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
//     if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
//     if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
//     if (tid < 32) warpReduce(sdata, tid);
//     if (tid == 0) g_odata[blockIdx.x] = sdata[0];
// }

__global__
void gpu_copy(float* src, float* dest, const uint32_t N){
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= N) return;
    
    dest[idx] = src[idx];
}
__global__ 
void Scale(float* v, float* s, float* v_acc uint32_t N){                // Used for calculating Ap in PCG
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= N) return;
    v_acc[idx] = *s * v[idx];
}

__global__
void sum_residuals(float* Ap, float* residual_dot_p, float J[N][M], uint32_t N, uint32_t M){  // Sums over r(i) with scalar vector product r(i)^T * p
    uint32_t idx = blockIdx.x * blockdim.x + threadIdx.x;
    if(idx >= N*M) return;
    uint32_t x = idx % N;
    uint32_t y = idx / N;
    atomicAdd(&Ap[x], residual_dot_p[y] * J[x][y]);
}

__global__
void scalar_divide(float* numerator, float* denominator, float* quotient){
    quotient = numerator / denominator;
}

__global__
void next_x(float* x, float* p, float* alpha, uint32_t N){
    uint32_t idx = blockIdx.x * blockdim.x + threadIdx.x;
    if(idx >= N) return;
    x[idx] = x[idx] + *alpha * p[idx];
}


__global__
void next_p(float* z, float* p, float* beta, uint32_t N){
    uint32_t idx = blockIdx.x * blockdim.x + threadIdx.x;
    if(idx >= N) return;
    p[idx] = z[idx] + *beta * p[idx];
}

__global__
void next_r(float* r, float* Ap, float* alpha, uint32_t N){
    uint32_t idx = blockIdx.x * blockdim.x + threadIdx.x;
    if(idx >= N) return;
    r[idx] = r[idx] - *alpha * Ap[idx];
} 
void PCG(float* J, float gamma, float alpha){
    // Move PCG algo here from GaussNewton below.



}

void AP( float* J, float* p, float* Ap, uint32_t N, uint32_t M){
    float* residual_dot_p;
    cudaMalloc((void **)&residual_dot_p, M * sizeof(float));
    dot<<<(N+255)/256, 256>>>(J, p, residual_dot_p, N);  // r(i)^T * p
    sum_residuals<<<((N*M)+255)/256, 256>>>(Ap, residual_dot_p, J, N, M);    // Ap = r(i) * (r(i)^T * p)
    cudaFree(residual_dot_p);
}



void GaussNewton::GaussNewtonUpdate(
    const float* x,   // Is named delta in init.py : Check argument position.
    const float* J,
    float* b,
    float* M_precon
    float gamma,
    float alpha,
    const bool* tiles_touched,
    const uint32_t N, // number of parameters
    const uint32_t M  // number of residuals
    ){
    
    float* M_precon;
    cudaMalloc((void **)&M_precon, N * sizeof(float));
    dim3 threadsPerBlock(256); // thread block size: 256
    dim3 numBlocks(((N*M)+255)/256); 

    diagJTJ<<<numBlocks, threadsPerBlock>>>(J, M_precon, N, M);
    
    // r_0 = b
    // z_0 = M^-1*r_0
    float* r= &b;
    float* dev_z;
    cudaMalloc((void **)&dev_z, N * sizeof(float));
    next_z<<<(N+255)/256, 256>>>(M_precon, b, dev_z, N);

    float eps = 0.1;
    float* h_R, h_R_prev;
    float* dev_R, dev_R_prev, dev_Ap;

    cudaMalloc((void **)&dev_R, sizeof(float));
    cudaMalloc((void **)&dev_R_prev, sizeof(float));
    cudaMalloc((void **)&dev_Ap, N * sizeof(float));

    h_R = (float *)malloc(sizeof(float));
    h_R_prev = (float *)malloc(sizeof(float));

    dot<<<(N+255)/256, 256>>>(b, b, dev_R_prev, N); // R_prev = rTr


    float* dev_p;
    cudaMalloc((void **)&dev_p, N * sizeof(float));
    gpu_copy<<<(N+255)/256, 256>>>(z, dev_p, N);


    float* dev_alpha;
    float* dev_denominator;
    float* dev_numerator;
    float* dev_beta;
//    float* host_alpha;
//    float* host_alpha_denominator;
//    float* host_alpha_numerator;
    cudaMalloc((void **)&dev_alpha, sizeof(float));
    cudaMalloc((void **)&dev_denominator, sizeof(float));
    cudaMalloc((void **)&dev_numerator, sizeof(float));
    cudaMalloc((void **)&dev_beta, sizeof(float));




    int k = 0; 
    const int MAX_ITERATIONS = 30;
    cudaMemcpy(h_R_prev, dev_R_prev, sizeof(float), cudaMemcpyDeviceToHost);

    while(k < MAX_ITERATIONS){
        dot<<<(N+255)/256, 256>>>(r, dev_z, dev_numerator, N); // r(k)^T * z(k)
        AP(J, dev_p, dev_Ap, N, M); 
        dot<<<(N+255)/256, 256>>>(dev_Ap, dev_p, dev_denominator, N); // p(k)^T * Ap(k) 
        scalar_divide<<<1, 1>>>(dev_numerator, dev_denominator, dev_alpha); // alpha = r(k)^T * z(k) / p(k)^T * Ap(k) 

        //Calculate next x
        next_x<<<(N+255)/256, 256>>>(x, dev_p, dev_alpha, N);
        dot<<<(N+255)/256, 256>>>(r, dev_z, dev_numerator, N); // r(k+1)^T * r(k+1), this is for the denomiator in the beta calculation
        next_r<<<(N+255)/256, 256>>>(r, dev_Ap, dev_alpha, N);
        dot<<<(N+255)/256, 256>>>(r, r, dev_R, N); // R = r(k+1)^T * r(k+1)

        // Check if R/Rprev > 0.85 or R < eps
        cudaMemcpy(h_R, dev_R, sizeof(float), cudaMemcpyDeviceToHost);
        if (*h_R/ *h_R_prev > 0.85 || *h_R < eps){
            break;
        }

        // 
        *h_R_prev = *h_R;
        next_z<<<(N+255)/256, 256>>>(M_precon, r, dev_z, N);
        dot<<<(N+255)/256, 256>>>(r, dev_z, dev_numerator, N); // r(k+1)^T * z(k+1)
        scalar_divide<<<1, 1>>>(dev_numerator, dev_denominator, dev_beta); // beta = r(k+1)^T * z(k+1) / r(k)^T * z(k)

        // Calculate next p
        next_p<<<(N+255)/256, 256>>>(dev_z, dev_p, dev_beta, N);
        k++;

    }






    //Free memory
    free(h_R);
    free(h_R_prev);

    cudaFree(dev_p);
    cudaFree(dev_R);
    cudaFree(dev_R_prev);
    cudaFree(dev_Ap);
    cudaFree(dev_alpha);
    cudaFree(dev_denominator);
    cudaFree(dev_numerator);
    cudaFree(dev_beta);
    cudaFree(dev_z);
    cudaFree(M_precon);
    cudaFree(r);



}
