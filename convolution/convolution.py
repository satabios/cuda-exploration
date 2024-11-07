import numpy as np
import torch
import torch.nn.functional as F



def numpy_conv2d(input_data, kernel_weights, padding=0, stride=1):
    if padding > 0:
        input_data = np.pad(input_data, 
                            ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                            mode='constant', constant_values=0)
    
    batch_size = input_data.shape[0]
    input_channels, output_channels = kernel_weights.shape[1], kernel_weights.shape[0]
    kernel_height, kernel_width = kernel_weights.shape[2], kernel_weights.shape[3]
    input_height, input_width = input_data.shape[2], input_data.shape[3]

    output_height = int((input_height - kernel_height) / stride) + 1
    output_width = int((input_width - kernel_width) / stride) + 1

    output_data = np.zeros((batch_size, output_channels, output_height, output_width))

    for output_channel in range(output_channels):
        kernel_data = kernel_weights[output_channel, :, :, :]
        for height in range(output_height):
            for width in range(output_width):
                region = input_data[:, :, height * stride:height * stride + kernel_height,
                                    width * stride:width * stride + kernel_width]
                output_data[:, output_channel, height, width] = np.sum(
                    region * kernel_data, axis=(1, 2, 3)
                )
    return output_data

def pytorch_conv2d(input_data, kernel_weights, kernel_bias, padding=0, stride=1):
    if padding > 0:
        input_data = F.pad(input_data, (padding, padding, padding, padding), "constant", 0)

    batch_size = input_data.shape[0]
    input_channels, output_channels = kernel_weights.shape[1], kernel_weights.shape[0]
    kernel_height, kernel_width = kernel_weights.shape[2], kernel_weights.shape[3]
    input_height, input_width = input_data.shape[2], input_data.shape[3]

    output_height = int((input_height - kernel_height) / stride) + 1
    output_width = int((input_width - kernel_width) / stride) + 1

    output_data = torch.zeros((batch_size, output_channels, output_height, output_width))

    for out_channel in range(output_channels):
        kernel_data = kernel_weights[out_channel, :, :, :]
        for height in range(output_height):
            for width in range(output_width):
                region = input_data[:, :, height * stride:height * stride + kernel_height,
                                    width * stride:width * stride + kernel_width]
                output_data[:, out_channel, height, width] = torch.sum(
                    region * kernel_data, dim=(1, 2, 3)
                )
            
    return output_data + kernel_bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # Expanding along Batch, Height and Width

def pytorch_linear(input_data, kernel_weights, kernel_bias):
    return torch.matmul(input_data, kernel_weights.T)+ kernel_bias.unsqueeze(0)
    

def pytorch_softmax(input_data, dim= -1):
    # As Large values of input may lead to overflow;
    # We subtract the largest value out of the input data
    # Before applying Softmax.
    # Note: Subtracting the same value from all the input data
    #       does not change the softmax result!
    input_max = torch.max(input_data, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(input_data- input_max)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x/sum_exp_x

def pytorch_maxpool2d(input_data, kernel_size, stride=0, padding=0):
    if padding > 0:
        input_data = torch.nn.functional.pad(input_data, (padding, padding, padding, padding))
    
    stride = kernel_size if stride==0 else stride
    batch_size, channels = input_data.shape[0], input_data.shape[1]
    height, width = input_data.shape[2], input_data.shape[3]
    kernel_height = (height-kernel_size+2*padding)//stride+1
    kernel_width = (width-kernel_size+2*padding)//stride+1
    output_data = torch.zeros(batch_size, channels, kernel_height, kernel_width)
    # Reoder the for loops depending on Row-Major or Columns-Major Memory Access
    for row in range(kernel_height):
        for col in range(kernel_width):
                output_data[:, :, row, col] = torch.amax(input_data[:, :, row*stride:row*stride+kernel_size, col*stride:col*stride+kernel_size], dim=(2, 3))
    
    return output_data
torch.manual_seed(4123)

if __name__ == "__main__":
    torcher = 1
    numpy = 0
    cuda = 0

    conv2d = 1
    linear = 1
    softmax = 1
    relu = 1
    maxpool2d = 1

    batch_size = 512
    if torcher:
        if(conv2d):
            # Add Dilation Feature
            input_data = torch.rand(batch_size, 3, 32, 32)
            kernel_weights = torch.rand(8, 3, 3, 3)

            padding = 0
            stride = 1

            torch_kernel = torch.nn.Conv2d(3, 8, 3, padding=padding, stride=stride)
            torch_kernel.weight.data = kernel_weights
            torch_out = torch_kernel(input_data)

            kernel_bias = torch_kernel.bias
            data_out = pytorch_conv2d(input_data, kernel_weights, kernel_bias, padding=padding, stride=stride)

            print("------------------ Conv2d ---------------------------")
            print("Custom PyTorch Conv Output Shape:", data_out.shape)
            print("Torch Built-in Conv Output Shape:", torch_out.shape)
            print("Close Match:", torch.allclose(torch_out, data_out, atol=1e-4))

        if(linear):
            input_data = torch.rand(batch_size, 256)
            kernel_weights = torch.rand(64, 256)

            input_dim, output_dim = 256, 64

            torch_kernel = torch.nn.Linear(256, 64)
            torch_kernel.weight.data = kernel_weights
            torch_out = torch_kernel(input_data)

            kernel_bias = torch_kernel.bias
            data_out = pytorch_linear(input_data, kernel_weights, kernel_bias)

            print("------------------ Linear ---------------------------")
            print("Custom PyTorch Function Output Shape:", data_out.shape)
            print("Torch Built-in Function Output Shape:", torch_out.shape)
            print("Close Match:", torch.allclose(torch_out, data_out, atol=1e-4))
        
        if(relu):
            input_data = torch.rand(batch_size, 256)

            torch_kernel = torch.nn.ReLU()
            torch_out = torch_kernel(input_data)

            data_out = torch.clip(input_data,min=0)

            print("------------------ ReLU ---------------------------")
            print("Custom PyTorch Function Output Shape:", data_out.shape)
            print("Torch Built-in Function Output Shape:", torch_out.shape)
            print("Close Match:", torch.allclose(torch_out, data_out, atol=1e-4))

        if(softmax):
            input_data = torch.rand(batch_size, 256)
            dim = -1
            torch_kernel = torch.nn.Softmax(dim=dim)
            torch_out = torch_kernel(input_data)

            data_out = pytorch_softmax(input_data, dim)
            
            print("------------------ Softmax ---------------------------")
            print("Custom PyTorch Function Output Shape:", data_out.shape)
            print("Torch Built-in Function Output Shape:", torch_out.shape)
            print("Close Match:", torch.allclose(torch_out, data_out, atol=1e-4))

        if(maxpool2d):

            input_data = torch.rand(batch_size,3,32,32)
            kernel_size = 3
            torch_kernel = torch.nn.MaxPool2d(kernel_size)
            torch_out = torch_kernel(input_data)

            data_out = pytorch_maxpool2d(input_data,kernel_size)
            print("------------------ MaxPool2d ---------------------------")
            print("Custom PyTorch Conv Output Shape:", data_out.shape)
            print("Torch Built-in Conv Output Shape:", torch_out.shape)
            print("Close Match:", torch.allclose(torch_out, data_out, atol=1e-4))



    if numpy:
        if(conv2d):
            input_data = np.random.rand(16, 3, 32, 32)
            kernel_weights = np.random.rand(8, 3, 3, 3)

            padding = 0
            stride = 1

            data_out = numpy_conv2d(input_data, kernel_weights, padding=padding, stride=stride)
            print("Numpy Conv Output Shape:", data_out.shape)
