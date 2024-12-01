#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void tiledMatrixMultiply(float *A, float *B, float *C) {
    // Global Indexes
    unsigned int Gcol = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int Grow = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.0f;
    __shared__ float SharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float SharedB[TILE_SIZE][TILE_SIZE];

    // Load the Shared Data
    for (unsigned int tileIdx = 0; tileIdx < COLS_A / TILE_SIZE; tileIdx++) { 
        unsigned int arow = Grow; 
        unsigned int acol = tileIdx * TILE_SIZE + threadIdx.y;

        unsigned int bcol = Gcol; 
        unsigned int brow = tileIdx * TILE_SIZE + threadIdx.x;

        if (arow < ROWS_A && acol < COLS_A) {
            SharedA[threadIdx.x][threadIdx.y] = A[arow * COLS_A + acol];
        } else {
            SharedA[threadIdx.x][threadIdx.y] = 0.0f;
        }

        if (brow < ROWS_B && bcol < COLS_B) {
            SharedB[threadIdx.x][threadIdx.y] = B[brow * COLS_B + bcol];
        } else {
            SharedB[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += SharedA[threadIdx.x][k] * SharedB[k][threadIdx.y];
        }
        __syncthreads(); 
    }

    if (Grow < ROWS_A && Gcol < COLS_B) {
        C[Grow * COLS_B + Gcol] = sum;
    }
}

void tiled_kernel(float *d_A, float *d_B, float *d_C) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(ceil(ROWS_A / TILE_SIZE), ceil(COLS_B / TILE_SIZE));
    tiledMatrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
}

__global__ void naiveMatrixMultiply(float *A, float *B, float *C) {
    unsigned int Gcol = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int Grow = blockDim.x * blockIdx.x + threadIdx.x;

    if (Grow < ROWS_A && Gcol < COLS_B) {
        float sum = 0.0f;
        for (int k = 0; k < COLS_A; k++) {
            sum += A[Grow * COLS_A + k] * B[k * COLS_B + Gcol];
        }
        C[Grow * COLS_B + Gcol] = sum;
    }
}

void naive_kernel(float *d_A, float *d_B, float *d_C) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(ceil(ROWS_A / TILE_SIZE), ceil(COLS_B / TILE_SIZE));
    naiveMatrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
}