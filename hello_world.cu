#include<stdio.h>

__global__ void hello_world(){
    printf("ThreadIdx: %d, %d, %d: BlockIdx: %d, %d, %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

}

int main(){
    dim3 blockDim(2,2);
    dim3 gridDim(1,3);
    hello_world<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
}