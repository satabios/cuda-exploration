#include <cuda_runtime.h>
#include <iostream>

// Helper function for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
                  << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Initialize data
void initialize_data(float *data, int dim1, int dim2, int dim3, int dim4, int fill) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                for (int l = 0; l < dim4; l++) {
                    if (fill == -1) 
                        data[i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l] = i + j + k + l;
                    else 
                        data[i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l] = fill;
                }
            }
        }
    }
}

// Display data
void display_data(float *data, int dim1, int dim2, int dim3, int dim4) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                for (int l = 0; l < dim4; l++) {
                    std::cout << data[i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

// CUDA kernel for convolution
__global__ void conv2d_naive_kernel(float *input, float *kernel, float *output, 
                                    int batch_size, int channel, int height, int width,
                                    int kernel_cout, int kernel_cin, int kernel_height, int kernel_width,
                                    int padding, int stride, 
                                    int feature_out_height, int feature_out_width) {
                                        
    int w_out = blockIdx.x * blockDim.x + threadIdx.x; // Output column index
    int h_out = blockIdx.y * blockDim.y + threadIdx.y; // Output row index
    int n = blockIdx.z / kernel_cout; // Batch index
    int c_out = blockIdx.z % kernel_cout; // Output channel index

    if (w_out < feature_out_width && h_out < feature_out_height) {
        float value = 0.0f;

        for (int c_in = 0; c_in < channel; c_in++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int h_in = h_out * stride - padding + kh;
                    int w_in = w_out * stride - padding + kw;
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        value += input[n * channel * height * width + c_in * height * width + h_in * width + w_in] *
                                 kernel[c_out * kernel_cin * kernel_height * kernel_width + c_in * kernel_height * kernel_width + kh * kernel_width + kw];
                    }
                }
            }
        }

        output[n * kernel_cout * feature_out_height * feature_out_width + c_out * feature_out_height * feature_out_width + h_out * feature_out_width + w_out] = value;
    }
}

// Perform 2D convolution
void conv2d(float *input_data, float* kernel, float* output_data, 
            int batch_size, int channel, int height, int width,
            int kernel_cout, int kernel_height, int kernel_width,
            int padding, int stride, 
            int feature_out_height, int feature_out_width) {
    float *d_input, *d_kernel, *d_output;
    size_t input_size = batch_size * channel * height * width * sizeof(float);
    size_t kernel_size = kernel_cout * channel * kernel_height * kernel_width * sizeof(float);
    size_t output_size = batch_size * kernel_cout * feature_out_height * feature_out_width * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));

    CUDA_CHECK(cudaMemcpy(d_input, input_data, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel, kernel_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, output_size));

    dim3 threadBlock(16, 16);
    dim3 gridBlock((feature_out_width + threadBlock.x - 1) / 16,
                   (feature_out_height + threadBlock.y - 1)/ 16,
                    kernel_cout * batch_size);

    conv2d_naive_kernel<<<gridBlock, threadBlock>>>(d_input, d_kernel, d_output, 
                                                    batch_size, channel, height, width,
                                                    kernel_cout, channel, kernel_height, kernel_width,
                                                    padding, stride, feature_out_height, feature_out_width);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output_data, d_output, output_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_output));
}

// Main
int main() {
    int batch_size = 1, channel = 1, height = 5, width = 5;
    int kernel_cout = 2, kernel_cin = channel, kernel_width = 3, kernel_height = 3;
    int padding = 0, stride = 1;

    int feature_out_height = (height - kernel_height + 2 * padding) / stride + 1;
    int feature_out_width = (width - kernel_width + 2 * padding) / stride + 1;

    float data[batch_size * channel * height * width];
    float kernel[kernel_cout * kernel_cin * kernel_height * kernel_width];
    float output_feature[batch_size * kernel_cout * feature_out_height * feature_out_width];

    initialize_data(data, batch_size, channel, height, width, -1);
    initialize_data(kernel, kernel_cout, kernel_cin, kernel_height, kernel_width, -1);

    std::cout << "Kernel Map:\n";
    display_data(kernel, kernel_cout, kernel_cin, kernel_height, kernel_width);

    std::cout << "Feature Map:\n";
    display_data(data, batch_size, channel, height, width);

    conv2d(data, kernel, output_feature, 
           batch_size, channel, height, width,
           kernel_cout, kernel_height, kernel_width,
           padding, stride, feature_out_height, feature_out_width);



    std::cout << "Output Feature Map:\n";
    display_data(output_feature, batch_size, kernel_cout, feature_out_height, feature_out_width);

    return 0;
}
