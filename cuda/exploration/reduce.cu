#include <iostream>

#define BLOCK_DIM 4
#define GRID_DIM 2

__global__ void reduce_kernel(float* input, float* partialSums, unsigned int N) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + tid;

    printf("Block %d, thread %d, i %d, Input:%f \n ", blockIdx.x, tid, i, input[i]);


    // Each thread performs a reduction within a block
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0 && i + stride < N) {
            input[i] += input[i + stride];
        }
        __syncthreads(); // Synchronize threads within the block
    }

    // Write the reduced value to partial sums
    if (tid == 0) {
        partialSums[blockIdx.x] = input[i];
    }
}

int main() {
    const unsigned int N = 4 * 4;  // Size of input array
    float* input = (float*)malloc(N * sizeof(float));
    float* partialSums = (float*)malloc(GRID_DIM * sizeof(float));

    // Initialize the input array with random values
    for (unsigned int i = 0; i < N; i++) {
        input[i] = i;
    }

    // Allocate device memory
    float* d_input;
    float* d_partialSums;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_partialSums, GRID_DIM * sizeof(float));

    // Copy input data to the device
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the reduction kernel
    reduce_kernel<<<GRID_DIM, BLOCK_DIM>>>(d_input, d_partialSums, N);

    // Copy the partial sums back to the host
    cudaMemcpy(partialSums, d_partialSums, GRID_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    // Perform a final reduction on the partial sums on the host
    float finalSum = 0.0f;
    for (unsigned int i = 0; i < GRID_DIM; i++) {
        finalSum += partialSums[i];
    }

    // Print the result
    printf("Final sum: %f\n", finalSum);

    // Free memory
    free(input);
    free(partialSums);
    cudaFree(d_input);
    cudaFree(d_partialSums);

    return 0;
}
