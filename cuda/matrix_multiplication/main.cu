#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cpu_kernel.h"
#include "gpu_kernel.cuh"
#include "test.h"


int main(){

    size_t sizeA = sizeof(float) * (ROWS_A * COLS_A);
    size_t sizeB = sizeof(float) * (ROWS_B * COLS_B);
    size_t sizeC = sizeof(float) * (ROWS_A * COLS_B);

    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C_Naive = (float *)malloc(sizeC);
    float *h_C_Tiled = (float *)malloc(sizeC);
    float *h_C_Tiled_Coarse = (float *)malloc(sizeC);
    float *h_C_ref = (float *)malloc(sizeC);

    for(int idx = 0; idx < ROWS_A * COLS_A; idx++) h_A[idx] = idx;
    for(int idx = 0; idx < ROWS_B * COLS_B; idx++) h_B[idx] = idx;
    
    float *d_A, *d_B, *d_C_Tiled, *d_C_Naive, *d_C_Tiled_Coarse;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C_Naive, sizeC);
    cudaMalloc(&d_C_Tiled, sizeC);
    cudaMalloc(&d_C_Tiled_Coarse, sizeC);
    
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    MatrixMulCPU(h_A, h_B, h_C_ref);
    naive_kernel(d_A, d_B, d_C_Naive);
    tiled_kernel(d_A, d_B, d_C_Tiled);
    tiled_coarse_kernel(d_A, d_B, d_C_Tiled_Coarse);

    cudaMemcpy(h_C_Naive, d_C_Naive, sizeC, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_Tiled, d_C_Tiled, sizeC, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_Tiled_Coarse, d_C_Tiled_Coarse, sizeC, cudaMemcpyDeviceToHost);

    testResult(h_C_Naive, h_C_ref, "Naive Kernel");
    testResult(h_C_Tiled, h_C_ref, "Tiled Kernel");
    testResult(h_C_Tiled_Coarse, h_C_ref, "Tiled Coarse Kernel");

    free(h_A);
    free(h_B);
    free(h_C_Naive);
    free(h_C_Tiled);
    free(h_C_Tiled_Coarse);
    free(h_C_ref); 

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_Naive);
    cudaFree(d_C_Tiled);
    cudaFree(d_C_Tiled_Coarse);

    return 0;

}