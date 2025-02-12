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
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N*M) return;
    uint32_t x = idx % N;
    uint32_t y = idx / N;


    float j_val = J[x + y * N];

    float tmp = j_val * j_val;

    atomicAdd(&M_precon[x],tmp);
}

__global__ void next_z(float* M_precon, float* r_0, float* z, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    float denominator = M_precon[idx];
    float inv = 1.0 / denominator;
    if (denominator == 0.0f){
        inv = 1.0;
    }
    float z_val = inv * r_0[idx];
    z[idx] = z_val;
}

// __global__ void rTr(float* r, float* R, uint32_t N){
//     float r_val = r[N];
//     atomicAdd(R, r_val*r_val);
// }

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

// void dot(const float* a, const float* b, float* c, uint32_t N) {
//     shared float temp[THREADS_PER_BLOCK];
//     uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

//     if (idx < N) {
//         temp[threadIdx.x] = a[idx] * b[idx];
//     } else {
//         temp[threadIdx.x] = 0.0f;
//     }

//     syncthreads();

//     // Perform reduction within the block
//     for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
//         if (threadIdx.x < stride) {
//             temp[threadIdx.x] += temp[threadIdx.x + stride];
//         }
//         syncthreads();
//     }

//     // Write the result of the block reduction to the output array
//     if (threadIdx.x == 0) {
//         atomicAdd(c, temp[0]);
//     }
// }

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
    p[idx] = z[idx] + *beta * p[idx];
}

__global__
void next_r(float* r, float* Ap, float* alpha, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    float new_val = r[idx] - *alpha * Ap[idx];
    r[idx] = new_val;
} 
void PCG(float* J, float gamma, float alpha){
    // Move PCG algo here from GaussNewton below.
    
    
    
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
    dot<<<(N+255)/256, 256>>>(J, v, residual_dot_v, N);  // r(i)^T * p
    sum_residuals<<<((N*M)+255)/256, 256>>>(Av, residual_dot_v, J, N, M);    // Ap = r(i) * (r(i)^T * p)
    cudaFree(residual_dot_v);
}


__global__ 
void subtract(float* a, float* b, float* c, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    c[idx] = a[idx] - b[idx];
}


void GaussNewton::gaussNewtonUpdate(
    float* x,   // Is named delta in init.py : Check argument position.
    float* J,
    float* b,
    float gamma,
    float alpha,
    const bool* tiles_touched,
    const uint32_t N, // number of parameters
    const uint32_t M  // number of residuals
    ){

    // printf("Address of b is %p\n", (void *)b);
    // printf("Address of x is %p\n", (void *)x);
    float* dev_r;
    cudaMalloc(&dev_r, N * sizeof(float));
    gpu_copy<<<(N+255)/256, 256>>>(b, dev_r, N);

    float* dev_Ax0;
    cudaMalloc(&dev_Ax0, N * sizeof(float));
    Av(J, x, dev_Ax0, N, M);

    subtract<<<(N+255)/256, 256>>>(dev_r, dev_Ax0, dev_r, N);
    
    float* M_precon;
    cudaMalloc(&M_precon, N * sizeof(float));
    dim3 threadsPerBlock(256); // thread block size: 256
    dim3 numBlocks(((N*M)+255)/256); 
    
    diagJTJ<<<numBlocks, threadsPerBlock>>>(J, M_precon, N, M);
    
    float* h_float;
    h_float = (float *)malloc(sizeof(float));
    // r_0 = b
    // z_0 = M^-1*r_0
    // float* r= b;

    // cudaMemcpy(h_float, &dev_r[0], sizeof(float), cudaMemcpyDeviceToHost);
    // printf("dev_r(b): %g \n", *h_float);

    float* dev_z;
    cudaMalloc(&dev_z, N * sizeof(float));
    next_z<<<(N+255)/256, 256>>>(M_precon, dev_r, dev_z, N);
    
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
    
    dot<<<(N+255)/256, 256>>>(dev_r, dev_r, dev_R_prev, N); // R_prev = rTr
    
    
    float* dev_p;
    cudaMalloc(&dev_p, N * sizeof(float));
    gpu_copy<<<(N+255)/256, 256>>>(dev_z, dev_p, N);
    
    
    float* dev_alpha;
    float* dev_denominator;
    float* dev_numerator;
    float* dev_beta;
    //    float* host_alpha;
    //    float* host_alpha_denominator;
    //    float* host_alpha_numerator;
    cudaMalloc(&dev_alpha, sizeof(float));
    cudaMalloc(&dev_denominator, sizeof(float));
    cudaMalloc(&dev_numerator, sizeof(float));
    cudaMalloc(&dev_beta, sizeof(float));

    
    
    
    
    int k = 0; 
    const int MAX_ITERATIONS = 5;
    cudaMemcpy(h_R_prev, dev_R_prev, sizeof(float), cudaMemcpyDeviceToHost);


    
    while(k < MAX_ITERATIONS){

        // float h_numerator = 1.0f;
        // float h_denominator = 2.0f;
        // cudaMemcpy(dev_numerator, &h_numerator, sizeof(float), cudaMemcpyHostToDevice);
        // cudaMemcpy(h_float, dev_numerator, sizeof(float), cudaMemcpyDeviceToHost);
        // printf("dev_numerator test: %f \n", *h_float);

        // cudaMemcpy(dev_denominator, &h_denominator, sizeof(float), cudaMemcpyHostToDevice);
        // cudaMemcpy(h_float, dev_denominator, sizeof(float), cudaMemcpyDeviceToHost);
        // printf("dev_denominator test: %f \n", *h_float);

        // scalar_divide<<<1, 1>>>(dev_numerator, dev_denominator, dev_alpha);
        // cudaMemcpy(h_float, dev_alpha, sizeof(float), cudaMemcpyDeviceToHost);
        // printf("dev_alpha test: %f \n", *h_float);

        // break;
        cudaMemset(dev_alpha, 0, sizeof(float));
        cudaMemset(dev_denominator, 0, sizeof(float));
        cudaMemset(dev_numerator, 0, sizeof(float));
        cudaMemset(dev_beta, 0, sizeof(float));


        // cudaMemcpy(h_float, dev_z, sizeof(float), cudaMemcpyDeviceToHost);
        // printf("dev_z test: %g \n", *h_float);
        dot<<<(N+255)/256, 256>>>(dev_r, dev_z, dev_numerator, N); // r(k)^T * z(k)
        // cudaMemcpy(h_float, dev_numerator, sizeof(float), cudaMemcpyDeviceToHost);
        // printf("dev_numerator test: %g \n", *h_float);

        Av(J, dev_p, dev_Ap, N, M); 

        dot<<<(N+255)/256, 256>>>(dev_p, dev_Ap, dev_denominator, N); // p(k)^T * Ap(k) 
        // cudaMemcpy(h_float, dev_denominator, sizeof(float), cudaMemcpyDeviceToHost);
        // printf("dev_denominator test: %g \n", *h_float);

        scalar_divide<<<1, 1>>>(dev_numerator, dev_denominator, dev_alpha); // alpha = r(k)^T * z(k) / p(k)^T * Ap(k) 
        
        //Calculate next x
        // float temp_alpha = 0.2;
        // cudaMemcpy(h_float, dev_alpha, sizeof(float), cudaMemcpyDeviceToHost);
        // printf("dev_alpha: %g \n", *h_float);

        // break;
        next_x<<<(N+255)/256, 256>>>(x, dev_p, dev_alpha, N);
        dot<<<(N+255)/256, 256>>>(dev_r, dev_z, dev_denominator, N); // r(k)^T * z(k), this is for the denomiator in the beta calculation
        // cudaMemcpy(h_float, &dev_r[0], sizeof(float), cudaMemcpyDeviceToHost);
        // printf("dev_r: %g \n", *h_float);
        // cudaMemcpy(h_float, &dev_Ap[0], sizeof(float), cudaMemcpyDeviceToHost);
        // printf("dev_Ap: %g \n", *h_float);
        next_r<<<(N+255)/256, 256>>>(dev_r, dev_Ap, dev_alpha, N);
        dot<<<(N+255)/256, 256>>>(dev_r, dev_r, dev_R, N); // R = r(k+1)^T * r(k+1)
        
        // Check if R/Rprev > 0.85 or R < eps
        cudaMemcpy(h_R, dev_R, sizeof(float), cudaMemcpyDeviceToHost);
        // printf("R: %g, Rprev: %g \n", *h_R, *h_R_prev);
        // printf("R/R_prev: %g \n", (*h_R/ *h_R_prev));
        if (*h_R/ *h_R_prev > 0.85f){
            break;
        }
        
        // if (*h_R < eps){
        //     break;
        // }

        // 
        *h_R_prev = *h_R;
        // printf("R_prev: %g \n", *h_R_prev);
        next_z<<<(N+255)/256, 256>>>(M_precon, dev_r, dev_z, N);
        dot<<<(N+255)/256, 256>>>(dev_r, dev_z, dev_numerator, N); // r(k+1)^T * z(k+1)
        scalar_divide<<<1, 1>>>(dev_numerator, dev_denominator, dev_beta); // beta = r(k+1)^T * z(k+1) / r(k)^T * z(k)
        // break;

        // Calculate next p
        next_p<<<(N+255)/256, 256>>>(dev_z, dev_p, dev_beta, N);
        k++;

    }

    // printf("end of GN PCG \n");




    //Free memory
    free(h_R);
    free(h_R_prev);
    free(h_float);

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
    cudaFree(dev_r);



}
