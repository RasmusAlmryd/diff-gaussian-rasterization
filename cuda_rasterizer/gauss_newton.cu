#include "auxiliary.h"
#include "gauss_newton.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;



__global__
void createJ(){

}



void PCG(float* J, float gamma, float alpha){
    r0 = ...

}


void GaussNewton::GaussNewtonUpdate(
    std::vector<float*> x,
    std::vector<const float*> J,
    float gamma,
    float alpha,
    const bool* tiles_touched,
    const uint32_t N,
    const uint32_t M){

    J = 

}
