import numpy as np
import torch


# Numpy implementation
# input_data = np.random.rand(16,3,32,32)
# kernel_weights = np.random.rand(8,3,3,3)
 
# output_data = np.zeros((batch_size, output_channels, output_height, output_width))

# for output_channel in range(output_channels):
#     for height in range(output_height):
#         for width in range(output_width):
#             output_data[:,output_channel,height,width] = np.sum(input_data[:,:,height:height+kernel_height,width:width+kernel_width] * kernel_weights[output_channel,:,:,:], axis=(1,2,3))
  

input_data = torch.rand(16,3,32,32)
kernel_weights = torch.rand(8,3,3,3)

padding = 0
stride = 1
batch_size  = input_data.shape[0]
input_channels, output_channels = kernel_weights.shape[1], kernel_weights.shape[0]
kernel_height, kernel_width = kernel_weights.shape[2], kernel_weights.shape[3]
input_height, input_width = input_data.shape[2], input_data.shape[3]

output_height = int((input_height - kernel_height + 2 * padding )/stride) + 1
output_width = int((input_width - kernel_width + 2 * padding )/stride) + 1
 
# Pytorch Implementation
output_data = torch.zeros((batch_size, output_channels, output_height, output_width))

for out_channel  in range(output_channels):
    kernel_data = kernel_weights[out_channel,:,:,:]
    for height in range(output_height):
        for width in range(output_width):
            output_data[:,out_channel,height, width] = torch.sum(input_data[:,:,height:height+kernel_height,width:width+kernel_width] * kernel_data, dim=(1,2,3))
print(output_data.shape)

