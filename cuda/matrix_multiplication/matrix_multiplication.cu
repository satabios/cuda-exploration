#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define ROWS_A 1024
#define COLS_A 1024
#define ROWS_B 1024
#define COLS_B 1024
#define TILE_SIZE 16

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplyKernel(float *a, float *b, float *c, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++) {
            sum += a[row * colsA + k] * b[k * colsB + col];
        }
        c[row * colsB + col] = sum;
    }
}

// CUDA kernel for matrix multiplication with shared memory tiling
__global__ void matrixMultiplyTiledKernel(float *a, float *b, float *c, int rowsA, int colsA, int colsB) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over the tiles of the input matrices
    for (int tileIdx = 0; tileIdx < (colsA + TILE_SIZE - 1) / TILE_SIZE; ++tileIdx) {
        // Load elements into shared memory
        tileA[threadIdx.y][threadIdx.x] = (row < rowsA && tileIdx * TILE_SIZE + threadIdx.x < colsA) ? a[row * colsA + tileIdx * TILE_SIZE + threadIdx.x] : 0.0f;
        
        tileB[threadIdx.y][threadIdx.x] = (col < colsB && tileIdx * TILE_SIZE + threadIdx.y < colsA) ? b[(tileIdx * TILE_SIZE + threadIdx.y) * colsB + col] : 0.0f;

        __syncthreads();  // Ensure all threads have loaded their elements

        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();  // Synchronize before loading the next tile
    }

    // Write the computed value to the result matrix
    if (row < rowsA && col < colsB) {
        c[row * colsB + col] = sum;
    }
}

int main() {
    // Host matrices
    float *a = new float[ROWS_A * COLS_A];
    float *b = new float[ROWS_B * COLS_B];
    float *c_naive = new float[ROWS_A * COLS_B];
    float *c_tiled = new float[ROWS_A * COLS_B];

    // Initialize host matrices with some values
    for (int i = 0; i < ROWS_A * COLS_A; i++) {
        a[i] = i;
    }
    for (int i = 0; i < ROWS_B * COLS_B; i++) {
        b[i] = i+6;
    }

    // Device pointers
    float *d_a, *d_b, *d_c;
    size_t sizeA = ROWS_A * COLS_A * sizeof(float);
    size_t sizeB = ROWS_B * COLS_B * sizeof(float);
    size_t sizeC = ROWS_A * COLS_B * sizeof(float);

    // Allocate device memory
    cudaMalloc((void **)&d_a, sizeA);
    cudaMalloc((void **)&d_b, sizeB);
    cudaMalloc((void **)&d_c, sizeC);

    // Copy data from host to device
    cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeB, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((COLS_B + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ROWS_A + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Timing for naive kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, ROWS_A, COLS_A, COLS_B);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_naive = 0;
    cudaEventElapsedTime(&milliseconds_naive, start, stop);

    // Copy result back to host for naive kernel
    cudaMemcpy(c_naive, d_c, sizeC, cudaMemcpyDeviceToHost);

    // Define block and grid sizes for tiled kernel
    dim3 threadsPerBlock_shared(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid_shared((COLS_B + TILE_SIZE - 1) / TILE_SIZE, 
                              (ROWS_A + TILE_SIZE - 1) / TILE_SIZE);

    // Timing for tiled kernel
    cudaEventRecord(start);
    matrixMultiplyTiledKernel<<<blocksPerGrid_shared, threadsPerBlock_shared>>>(d_a, d_b, d_c, ROWS_A, COLS_A, COLS_B);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_tiled = 0;
    cudaEventElapsedTime(&milliseconds_tiled, start, stop);

    // Copy result back to host for tiled kernel
    cudaMemcpy(c_tiled, d_c, sizeC, cudaMemcpyDeviceToHost);

    // Print timing results
    cout << "Naive Matrix Multiplication Time: " << milliseconds_naive << " ms" << endl;
    cout << "Tiled Matrix Multiplication Time: " << milliseconds_tiled << " ms" << endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    delete[] a;
    delete[] b;
    delete[] c_naive;
    delete[] c_tiled;

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
