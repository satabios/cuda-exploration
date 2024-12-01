// #include <cuda_runtime.h>
// #include <iostream>

// // Helper function for CUDA error checking
// #define CUDA_CHECK(call) do { \
//     cudaError_t err = call; \
//     if (err != cudaSuccess) { \
//         std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
//                   << ": " << cudaGetErrorString(err) << std::endl; \
//         exit(EXIT_FAILURE); \
//     } \
// } while (0)

// // Initialize data
// void initialize_data(float *data, int dim1, int dim2, int dim3, int dim4, int fill) {
//     for (int i = 0; i < dim1; i++) {
//         for (int j = 0; j < dim2; j++) {
//             for (int k = 0; k < dim3; k++) {
//                 for (int l = 0; l < dim4; l++) {
//                     if (fill == -1) 
//                         data[i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l] = i + j + k + l;
//                     else 
//                         data[i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l] = fill;
//                 }
//             }
//         }
//     }
// }

// // Display data
// void display_data(float *data, int dim1, int dim2, int dim3, int dim4) {
//     for (int i = 0; i < dim1; i++) {
//         for (int j = 0; j < dim2; j++) {
//             for (int k = 0; k < dim3; k++) {
//                 for (int l = 0; l < dim4; l++) {
//                     std::cout << data[i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l] << " ";
//                 }
//                 std::cout << std::endl;
//             }
//             std::cout << std::endl;
//         }
//         std::cout << std::endl;
//     }
// }

// // Naive CUDA kernel (kept from previous implementation)
// __global__ void conv2d_naive_kernel(float *input, float *kernel, float *output, 
//                                     int batch_size, int channel, int height, int width,
//                                     int kernel_cout, int kernel_cin, int kernel_height, int kernel_width,
//                                     int padding, int stride, 
//                                     int feature_out_height, int feature_out_width) {
                                        
//     int w_out = blockIdx.x * blockDim.x + threadIdx.x; // Output column index
//     int h_out = blockIdx.y * blockDim.y + threadIdx.y; // Output row index
//     int n = blockIdx.z / kernel_cout; // Batch index
//     int c_out = blockIdx.z % kernel_cout; // Output channel index

//     if (w_out < feature_out_width && h_out < feature_out_height) {
//         float value = 0.0f;

//         for (int c_in = 0; c_in < channel; c_in++) {
//             for (int kh = 0; kh < kernel_height; kh++) {
//                 for (int kw = 0; kw < kernel_width; kw++) {
//                     int h_in = h_out * stride - padding + kh;
//                     int w_in = w_out * stride - padding + kw;
//                     if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
//                         value += input[n * channel * height * width + c_in * height * width + h_in * width + w_in] *
//                                  kernel[c_out * kernel_cin * kernel_height * kernel_width + c_in * kernel_height * kernel_width + kh * kernel_width + kw];
//                     }
//                 }
//             }
//         }

//         output[n * kernel_cout * feature_out_height * feature_out_width + c_out * feature_out_height * feature_out_width + h_out * feature_out_width + w_out] = value;
//     }
// }

// // Shared Memory Tiling CUDA Kernel
// __global__ void conv2d_shared_memory_kernel(float *input, float *kernel, float *output, 
//                                             int batch_size, int channel, int height, int width,
//                                             int kernel_cout, int kernel_cin, int kernel_height, int kernel_width,
//                                             int padding, int stride, 
//                                             int feature_out_height, int feature_out_width) {
//     // Shared memory dimensions
//     const int TILE_WIDTH = 16;
//     const int TILE_HEIGHT = 16;
//     const int SHARED_MEM_RADIUS = (kernel_width - 1) / 2;
//     const int BLOCK_INPUT_WIDTH = TILE_WIDTH + 2 * SHARED_MEM_RADIUS;
//     const int BLOCK_INPUT_HEIGHT = TILE_HEIGHT + 2 * SHARED_MEM_RADIUS;

//     // Shared memory allocations
//     __shared__ float shared_input[BLOCK_INPUT_HEIGHT][BLOCK_INPUT_WIDTH];
//     __shared__ float shared_kernel[3][3];  // Assuming 3x3 kernel for this example

//     // Thread and block indices
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int n = blockIdx.z / kernel_cout;
//     int c_out = blockIdx.z % kernel_cout;

//     // Global output indices
//     int w_out = bx * TILE_WIDTH + tx;
//     int h_out = by * TILE_HEIGHT + ty;

//     // Load kernel into shared memory (only once by a few threads)
//     if (tx < kernel_height && ty < kernel_width && c_out == 0) {
//         shared_kernel[tx][ty] = kernel[c_out * kernel_cin * kernel_height * kernel_width + 
//                                         0 * kernel_height * kernel_width + tx * kernel_width + ty];
//     }
//     __syncthreads();

//     // Load input tiles into shared memory
//     int h_base = by * TILE_HEIGHT - SHARED_MEM_RADIUS;
//     int w_base = bx * TILE_WIDTH - SHARED_MEM_RADIUS;

//     // Each thread loads a single pixel
//     for (int c_in = 0; c_in < channel; c_in++) {
//         // Reset shared input to prevent data races
//         shared_input[ty + SHARED_MEM_RADIUS][tx + SHARED_MEM_RADIUS] = 0.0f;
//         __syncthreads();

//         // Load input data with boundary checking
//         int h_in = h_base + ty;
//         int w_in = w_base + tx;
//         if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
//             shared_input[ty + SHARED_MEM_RADIUS][tx + SHARED_MEM_RADIUS] = 
//                 input[n * channel * height * width + c_in * height * width + h_in * width + w_in];
//         }
//         __syncthreads();

//         // Convolution computation
//         if (w_out < feature_out_width && h_out < feature_out_height) {
//             float value = 0.0f;
//             for (int kh = 0; kh < kernel_height; kh++) {
//                 for (int kw = 0; kw < kernel_width; kw++) {
//                     value += shared_input[ty + kh][tx + kw] * 
//                              shared_kernel[kh][kw];
//                 }
//             }

//             // Accumulate output with multiple input channels
//             if (c_in == 0) {
//                 output[n * kernel_cout * feature_out_height * feature_out_width + 
//                        c_out * feature_out_height * feature_out_width + 
//                        h_out * feature_out_width + w_out] = value;
//             } else {
//                 output[n * kernel_cout * feature_out_height * feature_out_width + 
//                        c_out * feature_out_height * feature_out_width + 
//                        h_out * feature_out_width + w_out] += value;
//             }
//         }
//         __syncthreads();
//     }
// }

// // Perform 2D convolution with different kernel options
// void conv2d(float *input_data, float* kernel, float* output_data, 
//             int batch_size, int channel, int height, int width,
//             int kernel_cout, int kernel_height, int kernel_width,
//             int padding, int stride, 
//             int feature_out_height, int feature_out_width,
//             bool use_shared_memory = false) {
//     float *d_input, *d_kernel, *d_output;
//     size_t input_size = batch_size * channel * height * width * sizeof(float);
//     size_t kernel_size = kernel_cout * channel * kernel_height * kernel_width * sizeof(float);
//     size_t output_size = batch_size * kernel_cout * feature_out_height * feature_out_width * sizeof(float);

//     CUDA_CHECK(cudaMalloc(&d_input, input_size));
//     CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size));
//     CUDA_CHECK(cudaMalloc(&d_output, output_size));

//     CUDA_CHECK(cudaMemcpy(d_input, input_data, input_size, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_kernel, kernel, kernel_size, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemset(d_output, 0, output_size));

//     dim3 threadBlock(16, 16);
//     dim3 gridBlock((feature_out_width + threadBlock.x - 1) / 16,
//                    (feature_out_height + threadBlock.y - 1)/ 16,
//                     kernel_cout * batch_size);

//     // Choose kernel based on shared memory flag
//     if (use_shared_memory) {
//         conv2d_shared_memory_kernel<<<gridBlock, threadBlock>>>(d_input, d_kernel, d_output, 
//                                                                 batch_size, channel, height, width,
//                                                                 kernel_cout, channel, kernel_height, kernel_width,
//                                                                 padding, stride, feature_out_height, feature_out_width);
//     } else {
//         conv2d_naive_kernel<<<gridBlock, threadBlock>>>(d_input, d_kernel, d_output, 
//                                                         batch_size, channel, height, width,
//                                                         kernel_cout, channel, kernel_height, kernel_width,
//                                                         padding, stride, feature_out_height, feature_out_width);
//     }

//     CUDA_CHECK(cudaDeviceSynchronize());

//     CUDA_CHECK(cudaMemcpy(output_data, d_output, output_size, cudaMemcpyDeviceToHost));

//     CUDA_CHECK(cudaFree(d_input));
//     CUDA_CHECK(cudaFree(d_kernel));
//     CUDA_CHECK(cudaFree(d_output));
// }

// // Main
// int main() {
//     int batch_size = 1, channel = 1, height = 5, width = 5;
//     int kernel_cout = 2, kernel_cin = channel, kernel_width = 3, kernel_height = 3;
//     int padding = 0, stride = 1;

//     int feature_out_height = (height - kernel_height + 2 * padding) / stride + 1;
//     int feature_out_width = (width - kernel_width + 2 * padding) / stride + 1;

//     float data[batch_size * channel * height * width];
//     float kernel[kernel_cout * kernel_cin * kernel_height * kernel_width];
//     float output_feature_naive[batch_size * kernel_cout * feature_out_height * feature_out_width];
//     float output_feature_shared[batch_size * kernel_cout * feature_out_height * feature_out_width];

//     initialize_data(data, batch_size, channel, height, width, -1);
//     initialize_data(kernel, kernel_cout, kernel_cin, kernel_height, kernel_width, -1);

//     std::cout << "Kernel Map:\n";
//     display_data(kernel, kernel_cout, kernel_cin, kernel_height, kernel_width);

//     std::cout << "Feature Map:\n";
//     display_data(data, batch_size, channel, height, width);

//     // Naive Convolution
//     conv2d(data, kernel, output_feature_naive, 
//            batch_size, channel, height, width,
//            kernel_cout, kernel_height, kernel_width,
//            padding, stride, feature_out_height, feature_out_width, false);

//     // Shared Memory Convolution
//     conv2d(data, kernel, output_feature_shared, 
//            batch_size, channel, height, width,
//            kernel_cout, kernel_height, kernel_width,
//            padding, stride, feature_out_height, feature_out_width, true);

//     std::cout << "Output Feature Map (Naive):\n";
//     display_data(output_feature_naive, batch_size, kernel_cout, feature_out_height, feature_out_width);

//     std::cout << "Output Feature Map (Shared Memory):\n";
//     display_data(output_feature_shared, batch_size, kernel_cout, feature_out_height, feature_out_width);

//     return 0;
// }


#include <stdio.h>

__global__ void convolution_kernel(float* input, float* output, unsigned int width, unsigned int height) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    int MASK_DIM = 2;
    int MASK_RADIUS = MASK_DIM / 2;

    // Example 2x2 mask (kernel) - replace with your desired values
    float mask_c[2][2] = {
        {1.0f / 4.0f, 1.0f / 4.0f},
        {1.0f / 4.0f, 1.0f / 4.0f}
    };

    float sum = 0.0f;

    for (int maskRow = 0; maskRow < MASK_DIM; ++maskRow) {
        for (int maskCol = 0; maskCol < MASK_DIM; ++maskCol) {
            int inRow = outRow - MASK_RADIUS + maskRow;
            int inCol = outCol - MASK_RADIUS + maskCol;
            // printf("inRow:(%d) = outRow(%d) - MASK_RADIUS(%d) + maskRow(%d) \n", inRow, outRow, MASK_RADIUS, maskRow);
            printf("In: (%d, %d), Out: (%d, %d) \n", inRow, inCol, outRow, outCol );
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                sum += mask_c[maskRow][maskCol] * input[inRow * width + inCol];
            }
        }
    }

    output[outRow * width + outCol] = sum;
}

int main() {
    // Example usage:
    const int width = 4;
    const int height = 4;

    float input[width * height] = { 
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f 
    };
    float output[width * height];

    // Allocate device memory
    float* d_input, * d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(4, 4); // Example block size
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    convolution_kernel << <gridDim, blockDim >> > (d_input, d_output, width, height);

    // Copy output data from device to host
    cudaMemcpy(output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the output (for verification)
    printf("Output:\n");
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%.2f ", output[i * width + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}