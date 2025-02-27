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



void GaussNewton::gaussNewtonUpdate(
    float* x,   // Is named delta in init.py : Check argument position.
    float* sparse_J_values,
    uint64_t* sparse_J_indices,
    uint32_t* sparse_J_p_sum,
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

    // b = JTv (v = r) = JTr

    /*
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
    */

}


    // printing variable
    // float* h_float;
    // h_float = (float *)malloc(sizeof(float));
    // cudaMemcpy(h_float, &x[0], sizeof(float), cudaMemcpyDeviceToHost);
    // printf("x(0): %g \n", *h_float);
