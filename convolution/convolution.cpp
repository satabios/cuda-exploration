#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

// Helper function to initialize random data
void initializeRandomData(MatrixXd& data, int total_size) {
    data = MatrixXd::Random(total_size, 1);
}

// Function to perform convolution operation
void convolution3D(const MatrixXd& input_data, const MatrixXd& kernel_weights, MatrixXd& output_data,
                   int batch_size, int input_channels, int output_channels,
                   int input_height, int input_width, int kernel_height, int kernel_width,
                   int output_height, int output_width, int padding, int stride) {
    // Initialize output_data
    output_data.resize(batch_size * output_channels * output_height * output_width, 1);
    output_data.setZero();

    // Perform convolution
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int out_channel = 0; out_channel < output_channels; ++out_channel) {
            for (int out_y = 0; out_y < output_height; ++out_y) {
                for (int out_x = 0; out_x < output_width; ++out_x) {
                    double sum = 0.0;
                    for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
                        for (int kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
                            for (int kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
                                int in_y = out_y * stride + kernel_y - padding;
                                int in_x = out_x * stride + kernel_x - padding;
                                if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                                    int input_index = batch * input_channels * input_height * input_width + in_channel * input_height * input_width + in_y * input_width + in_x;
                                    int kernel_index = out_channel * input_channels * kernel_height * kernel_width + in_channel * kernel_height * kernel_width + kernel_y * kernel_width + kernel_x;
                                    sum += input_data(input_index, 0) * kernel_weights(kernel_index, 0);
                                }
                            }
                        }
                    }
                    int output_index = batch * output_channels * output_height * output_width + out_channel * output_height * output_width + out_y * output_width + out_x;
                    output_data(output_index, 0) = sum;
                }
            }
        }
    }
}

int main() {
    int batch_size = 16;
    int input_channels = 3;
    int output_channels = 8;
    int input_height = 32;
    int input_width = 32;
    int kernel_height = 3;
    int kernel_width = 3;
    int padding = 0;
    int stride = 1;

    // Calculate output dimensions
    int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;

    // Initialize input data and kernel weights
    MatrixXd input_data(batch_size * input_channels * input_height * input_width, 1);
    MatrixXd kernel_weights(output_channels * input_channels * kernel_height * kernel_width, 1);
    MatrixXd output_data;

    // Populate input data and kernel weights with random values
    initializeRandomData(input_data, input_data.rows());
    initializeRandomData(kernel_weights, kernel_weights.rows());

    // Perform convolution
    convolution3D(input_data, kernel_weights, output_data, batch_size, input_channels, output_channels,
                  input_height, input_width, kernel_height, kernel_width, output_height, output_width, padding, stride);

    // Output result shape
    cout << "Output shape: (" << batch_size << ", " << output_channels << ", " << output_height << ", " << output_width << ")" << endl;

    return 0;
}
