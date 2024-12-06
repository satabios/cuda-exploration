#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_BINS 256
#define COARSENING_FACTOR 1

__global__ void histogram_kernel(const unsigned char* input, unsigned int* histogram, int size) { 
  __shared__ unsigned int local_histogram[NUM_BINS];

  // Initialize the local histogram
  for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    local_histogram[i] = 0;
  }
  __syncthreads();

  // Calculate local histogram with coarsening
  int global_start = blockIdx.x * blockDim.x * COARSENING_FACTOR + threadIdx.x;
  int stride = blockDim.x * gridDim.x * COARSENING_FACTOR;

  for (int i = global_start; i < size; i += stride) {
    for (int j = 0; j < COARSENING_FACTOR && i + j < size; ++j) {
      unsigned char value = input[i + j];
      atomicAdd(&local_histogram[value], 1);
    }
  }
  __syncthreads();

  // Accumulate local histogram into global histogram
  for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    atomicAdd(&histogram[i], local_histogram[i]);
  }
}

// __global__ void histogram_kernel(const unsigned char* input, unsigned int* histogram, int size) { 
//   __shared__ int local_histogram[256];

//   for (int i = threadIdx.x; i < 256; i += blockDim.x) {
//     local_histogram[i] = 0;
//   }
//   __syncthreads();

//   // Calculate local histogram
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;
//   while (i < size) {
//     // Privatization: Load the input value into a register
//     unsigned char value = input[i];

//     // Increment the corresponding bin in the local histogram
//     atomicAdd(&local_histogram[value], 1);

//     i += stride;
//   }
//   __syncthreads();

//   // Accumulate local histogram into global histogram
//   for (int i = threadIdx.x; i < 256; i += blockDim.x) {
//     atomicAdd(&histogram[i], local_histogram[i]);
//   }
// }
// __global__ void histogram_kernel(const unsigned char* input, unsigned int* histogram, int size) { 
//   __shared__ unsigned int local_histogram[256]; // Declare shared memory for local histogram

//   // Initialize the local histogram to zero
//   for (int i = threadIdx.x; i < 256; i += blockDim.x) {
//     local_histogram[i] = 0;
//   }
//   __syncthreads(); // Ensure all threads complete initialization before proceeding

//   // Calculate local histogram
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;

//   while (i < size) {
//     // Load the input value
//     unsigned char value = input[i];

//     // Increment the corresponding bin in the local histogram
//     atomicAdd(&local_histogram[value], 1);

//     i += stride;
//   }
//   __syncthreads(); // Ensure all threads finish updating the local histogram

//   // Accumulate local histogram into global histogram
//   for (int i = threadIdx.x; i < 256; i += blockDim.x) {
//     atomicAdd(&histogram[i], local_histogram[i]);
//   }
// }


// Helper function for error checking
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
    exit(1);
  }
}

int main() {
  // Image dimensions
  int width = 512;
  int height = 512;
  int size = width * height;

  // Allocate host memory for image and histogram
  unsigned char* image = (unsigned char*)malloc(size * sizeof(unsigned char));
  unsigned int* bins = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));

  // Initialize image data (example: random values)
  for (int i = 0; i < size; i++) {
    image[i] = rand() % 256;
  }

  // Allocate device memory
  unsigned char* d_image;
  unsigned int* d_bins;
  checkCudaErrors(cudaMalloc((void**)&d_image, size * sizeof(unsigned char)));
  checkCudaErrors(cudaMalloc((void**)&d_bins, NUM_BINS * sizeof(unsigned int)));

  // Copy data to device
  checkCudaErrors(cudaMemcpy(d_image, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice));

  // Initialize device memory for histogram to 0 using cudaMemset
  checkCudaErrors(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int))); 
  
  // --- Timing the kernel ---
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // Record start event
  checkCudaErrors(cudaEventRecord(start));

  // Launch kernel with increased threads per block
  int threadsPerBlock = 1024; // Higher thread count for coarsening
  int blocksPerGrid = (size + threadsPerBlock * COARSENING_FACTOR - 1) / (threadsPerBlock * COARSENING_FACTOR);
  histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_bins, size);
  checkCudaErrors(cudaDeviceSynchronize());

  // Record stop event
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  // Calculate elapsed time in milliseconds
  float milliseconds = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Kernel execution time: %.3f ms\n", milliseconds);
  // --- End of timing ---

  // Copy results back to host
  checkCudaErrors(cudaMemcpy(bins, d_bins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  // Free memory
  free(image);
  free(bins);
  checkCudaErrors(cudaFree(d_image));
  checkCudaErrors(cudaFree(d_bins));

  return 0;
}
