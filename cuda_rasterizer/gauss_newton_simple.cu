
#include "auxiliary.h"
#include "gauss_newton_simple.h"
// #include "backward.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#include <string>


#define THREADS_PER_BLOCK 256


namespace GNS_kernels{



__global__
void diagJTJ(
    const float* J, 
    float* M_precon, 
    const uint32_t N, 
    const uint32_t M
){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N*M) return;

    int x = idx % N;  // Parameter index
    int y = idx / N;  // Pixel index

    float res2 = J[y * N + x] * J[y * N + x];
    atomicAdd(&M_precon[x], res2);
}


__global__
void JTr(
    const float* J,
    const float* residuals,
    float* b,
    const uint32_t N, 
    const uint32_t M
){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N*M) return;

    int x = idx % N;  // Parameter index
    int y = idx / N;  // Pixel index

    atomicAdd(&b[x], -J[y * N + x] * residuals[y]);
}

__global__ void next_z(float* M_precon, float* r, float* z, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    float denominator = M_precon[idx];
    // float inv = 1.0 / denominator;
    // if (denominator == 0.0f){
    //     inv = 1.0;
    // }
    // float z_val = 0;
    // if (denominator != 0.0f){
        //     z_val = r[idx] / denominator;
        // }
    float inv = 1.0 / (denominator + 1e-8f);
    float z_val = inv * r[idx];
    z[idx] = z_val;
}

__global__ void dot(float *a, float *b, float *c, uint32_t N){
    __shared__ float temp[THREADS_PER_BLOCK];
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= N) return;

    temp[threadIdx.x] = a[idx] * b[idx];
    printf("temp[%d]: %g \n", idx, temp[idx]);

    __syncthreads();

    if (threadIdx.x == 0)
    {
        float sum = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++)
        {
            sum += temp[i];
        }
        printf("sum: %g", sum);
        atomicAdd(&c[0], sum);
    }
}

__global__
void residual_dot_sum(float* J, float* v, float* residual_dot_v, uint32_t N, uint32_t M){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N*M) return;
    int x = idx % N;  // Parameter index
    int y = idx / N;  // Pixel index
    float dot_sum = J[x + y * N] * v[x];
    // printf("(x: %d, y: %d) J*v: %g \n",x,y, dot_sum);
    atomicAdd(&residual_dot_v[y], dot_sum);
}


__global__
void sum_residuals(float* Ap, float* residual_dot_v, float* J, uint32_t N, uint32_t M){  // Sums over r(i) with scalar vector product r(i)^T * p
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N*M) return;
    int x = idx % N;  // Parameter index
    int y = idx / N;  // Pixel index
    // printf("(x: %d, y: %d) residual: %g, J: %g \n",x,y, residual_dot_v[y], J[x + y * N]);

    atomicAdd(&Ap[x], residual_dot_v[y] * J[x + y * N]);
}


void Av(float* J, float* v, float* Av, uint32_t N, uint32_t M){
    float* residual_dot_v;
    cudaMalloc(&residual_dot_v, M * sizeof(float));
    cudaMemset(residual_dot_v, 0, M * sizeof(float));
    residual_dot_sum<<<((N*M)+255)/256, 256>>>(J, v, residual_dot_v, N, M);    // (r(i)^T * p)
    sum_residuals<<<((N*M)+255)/256, 256>>>(Av, residual_dot_v, J, N, M);    // Ap = r(i) * (r(i)^T * p)
    cudaFree(residual_dot_v);
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

__global__ 
void subtract(float* a, float* b, float* c, uint32_t N){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    c[idx] = a[idx] - b[idx];
}

}

















void GaussNewtonSimple::gaussNewtonUpdate(
    float* x,   // Is named delta in init.py : Check argument position.
    float* J,  // stored in residual order hmmm
    float* residuals,
    const uint32_t N, // number of parameters
    const uint32_t M  // number of residuals
){
    
    float test_val;

    float* b;
    cudaMalloc(&b, N * sizeof(float));
    cudaMemset(b,0, N * sizeof(float));
    printf("tot num params: %d \n", N);
    printf("tot num residuals: %d \n", M);

    // b = -JTr
    GNS_kernels::JTr<<<(N*M+255)/256, 256>>>(
        J,
        residuals,
        b,
        N,
        M
    );

    for(int i = 0; i<N; i++){
        cudaMemcpy(&test_val, &b[i], sizeof(float), cudaMemcpyDeviceToHost);
        printf("b(%d): %g\n", i, test_val);
    }

    float* Ap;
    cudaMalloc(&Ap, N * sizeof(float));
    cudaMemset(Ap, 0, N * sizeof(float));

    GNS_kernels::Av(
        J,
        x,
        Ap,
        N,
        M
    );

    // r(0) = b - A*x(0)
    float* r;
    cudaMalloc(&r, N * sizeof(float));
    GNS_kernels::subtract<<<(N+255)/256, 256>>>(b, Ap, r, N);



    // M = diag(J^T * J)
    float* M_precon;
    cudaMalloc(&M_precon, N * sizeof(float));
    cudaMemset(M_precon, 0, N * sizeof(float));


    GNS_kernels::diagJTJ<<<((N*M)+255)/256, 256>>>(
        J, 
        M_precon, 
        N, 
        M
    );

    for(int i = 0; i<N; i++){
        cudaMemcpy(&test_val, &M_precon[i], sizeof(float), cudaMemcpyDeviceToHost);
        printf("M(%d): %g\n", i, test_val);
    }

    // z = M^-1*r
    float* z;
    cudaMalloc(&z, N * sizeof(float));
    GNS_kernels::next_z<<<(N+255)/256, 256>>>(M_precon, r, z, N);

    for(int i = 0; i<N; i++){
        cudaMemcpy(&test_val, &z[i], sizeof(float), cudaMemcpyDeviceToHost);
        printf("z(%d): %g\n", i, test_val);
    }


    // p = z
    float* p;
    cudaMalloc(&p, N * sizeof(float));
    cudaMemcpy(p, z, N * sizeof(float), cudaMemcpyDeviceToDevice);



    float eps = 1e-8;
    float h_R;
    float h_R_prev;
    float* dev_R;
    float* dev_R_prev;

    cudaMalloc(&dev_R, sizeof(float));
    cudaMalloc(&dev_R_prev, sizeof(float));
    cudaMemset(dev_R, 0, sizeof(float));
    cudaMemset(dev_R_prev, 0, sizeof(float));


    // R = rTr
    GNS_kernels::dot<<<(N+255)/256, 256>>>(r, r, dev_R_prev, N);


    float* alpha;
    float* denominator;
    float* numerator;
    float* beta;
    cudaMalloc(&alpha, sizeof(float));
    cudaMalloc(&numerator, sizeof(float));
    cudaMalloc(&denominator, sizeof(float));
    cudaMalloc(&beta, sizeof(float));


    int k = 0; 
    const int MAX_ITERATIONS = 5;
    cudaMemcpy(&h_R_prev, dev_R_prev, sizeof(float), cudaMemcpyDeviceToHost);

    while(k < MAX_ITERATIONS){

        
        cudaMemset(alpha, 0, sizeof(float));
        cudaMemset(denominator, 0, sizeof(float));
        cudaMemset(numerator, 0, sizeof(float));
        cudaMemset(beta, 0, sizeof(float));
        cudaMemset(Ap, 0, N * sizeof(float));

        // rTz
        GNS_kernels::dot<<<(N+255)/256, 256>>>(r, z, numerator, N);

        // Ap = JTJp
        GNS_kernels::Av(
            J, 
            p, 
            Ap, 
            N, 
            M
        );

        for(int i = 0; i<N; i++){
            cudaMemcpy(&test_val, &Ap[i], sizeof(float), cudaMemcpyDeviceToHost);
            printf("Ap(%d): %g\n", i, test_val);
        }


        // p(k)^T * Ap(k) 
        GNS_kernels::dot<<<(N+255)/256, 256>>>(p, Ap, denominator, N);

        // alpha = r(k)^T * z(k) / p(k)^T * Ap(k) 
        GNS_kernels::scalar_divide<<<1, 1>>>(numerator, denominator, alpha); 


        // x(k+1) = x + alpha * p
        GNS_kernels::next_x<<<(N+255)/256, 256>>>(x, p, alpha, N);
        for(int i = 0; i<N; i++){
            cudaMemcpy(&test_val, &x[i], sizeof(float), cudaMemcpyDeviceToHost);
            printf("x(%d): %g\n", i, test_val);
        }

        cudaMemcpy(&test_val, alpha, sizeof(float), cudaMemcpyDeviceToHost);
        printf("alpha: %g\n", test_val);
        
        // r(k)^T * z(k)  (used for beta calculation)
        cudaMemset(denominator, 0, sizeof(float));
        GNS_kernels::dot<<<(N+255)/256, 256>>>(r, z, denominator, N);



        // r(k+1) = r(k) - alpha * Ap(k)
        GNS_kernels::next_r<<<(N+255)/256, 256>>>(r, Ap, alpha, N);
        for(int i = 0; i<N; i++){
            cudaMemcpy(&test_val, &r[i], sizeof(float), cudaMemcpyDeviceToHost);
            printf("r(%d): %g\n", i, test_val);
        }



        // R = r(k+1)^T * r(k+1)
        cudaMemset(dev_R, 0, sizeof(float));
        cudaMemcpy(&test_val, dev_R, sizeof(float), cudaMemcpyDeviceToHost);
        printf("dev_R: %g\n", test_val);
        GNS_kernels::dot<<<(N+255)/256, 256>>>(r, r, dev_R, N); 
        
        // Check if R/Rprev > 0.85 or R < eps
        cudaMemcpy(&h_R, dev_R, sizeof(float), cudaMemcpyDeviceToHost);
        printf("R: %g, Rprev: %g \n", h_R, h_R_prev);
        printf("R/R_prev: %g \n", (h_R/ h_R_prev));
        if (h_R < eps || h_R/ h_R_prev > 0.85f){
            break;
        }
        
        h_R_prev = h_R;

        // z(k+1) = M^-1 * r(k)
        GNS_kernels::next_z<<<(N+255)/256, 256>>>(M_precon, r, z, N);

        for(int i = 0; i<N; i++){
            cudaMemcpy(&test_val, &z[i], sizeof(float), cudaMemcpyDeviceToHost);
            printf("z(%d): %g\n", i, test_val);
        }
        
        // r(k+1)^T * z(k+1)
        cudaMemset(numerator, 0, sizeof(float));
        GNS_kernels::dot<<<(N+255)/256, 256>>>(r, z, numerator, N); 

        cudaMemcpy(&test_val, numerator, sizeof(float), cudaMemcpyDeviceToHost);
        printf("numerator: %g\n", test_val);

        cudaMemcpy(&test_val, denominator, sizeof(float), cudaMemcpyDeviceToHost);
        printf("denominator: %g\n", test_val);


        
        // beta = r(k+1)^T * z(k+1) / r(k)^T * z(k)
        GNS_kernels::scalar_divide<<<1, 1>>>(numerator, denominator, beta); 
        cudaMemcpy(&test_val, beta, sizeof(float), cudaMemcpyDeviceToHost);
        printf("beta: %g\n", test_val);

        // p(k+1) = z(k) + beta * p(k)
        GNS_kernels::next_p<<<(N+255)/256, 256>>>(z, p, beta, N);

        for(int i = 0; i<N; i++){
            cudaMemcpy(&test_val, &p[i], sizeof(float), cudaMemcpyDeviceToHost);
            printf("p(%d): %g\n", i, test_val);
        }

        k++;
    }

    //Free memory
    // free(h_R);
    // free(h_R_prev);
    // free(h_float);

    cudaFree(p);
    cudaFree(dev_R);
    cudaFree(dev_R_prev);
    cudaFree(Ap);
    cudaFree(alpha);
    cudaFree(numerator);
    cudaFree(denominator);
    cudaFree(beta);
    cudaFree(z);
    cudaFree(M_precon);
    cudaFree(r);
    cudaFree(b);
}