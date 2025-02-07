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

__global__ void z_0(float* M_precon, float* r_0, float* z, uint32_t N){
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


void PCG(float* J, float gamma, float alpha){
    

}


void GaussNewton::GaussNewtonUpdate(
    const float* x,
    const float* J,
    float* b,
    float* M_precon
    float gamma,
    float alpha,
    const bool* tiles_touched,
    const uint32_t N, // number of parameters
    const uint32_t M  // number of residuals
    ){
    
    dim3 threadsPerBlock(256); // thread block size: 256
    dim3 numBlocks(((N*M)+255)/256); 

    diagJTJ<<<numBlocks, threadsPerBlock>>>(J, M_precon, N, M);
    
    // r_0 = b
    // z_0 = M^-1*r_0
    float* z = &M_precon;
    z_0<<<(N+255)/256, 256>>>(M_precon, b, z, N);

    float eps = 0.1
    float* h_R, h_R_prev;
    float* dev_R, dev_R_prev;

    cudaMalloc((void **)&dev_R, sizeof(float));
    cudaMalloc((void **)&dev_R_prev, sizeof(float));

    h_R = (float *)malloc(sizeof(float));
    h_R_prev = (float *)malloc(sizeof(float));

    dot<<<(N+255)/256, 256>>>(b, b, dev_R_prev); // R_prev = rTr


    float* dev_p;
    cudaMalloc((void **)&dev_p, N * sizeof(float));
    gpu_copy<<<(N+255)/256, 256>>>(z, dev_p);


    float* dev_alpha;
    cudaMalloc((void **)&dev_alpha, sizeof(float));
    
    int k = 0; 
    const int MAX_ITERATIONS = 30;

    while(k < MAX_ITERATIONS){
        dot<<<(N+255)/256, 256>>>(r, z, dev_alpha);
    }






    //Free memory
    free(h_R);
    free(h_R_prev);

    cudaFree(dev_p);
    cudaFree(dev_R);
    cudaFree(dev_R_prev);

}
