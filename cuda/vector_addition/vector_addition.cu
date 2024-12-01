#include<stdio.h>

#define BLOCK_SIZE 32
#define N 64

__global__ void vector_addition(int *a, int *b, int *c){
    
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N){
        printf("ThreadIdx: %d, BlockIdx: %d blockDim: %d\n", threadIdx.x, blockIdx.x, blockDim.x);
        c[i] = a[i] + b[i];
    }
}



int main(){
    

    int *a, *b, *c; // host variables
    int *d_a, *d_b, *d_c; // device variables
    // Memory allocation on host
    a = (int*)malloc(N*sizeof(int)); 
    b = (int*)malloc(N*sizeof(int));
    c = (int*)malloc(N*sizeof(int));
    // Memory allocation on device
    cudaMalloc(&d_a, N*sizeof(int));
    cudaMalloc(&d_b, N*sizeof(int));
    cudaMalloc(&d_c, N*sizeof(int));
    // Initialize host variables
    for(int i=0; i<N; i++){
        a[i] = i;
        b[i] = i;
    }
    // Copy host variables to device
    cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
    // Kernel launch
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((N+threadsPerBlock.x-1)/threadsPerBlock.x); // (16+16-1)/16 = 2 ; General Formula: (n+blockDim-1)/blockDim for number that are not multiple of blockDim
    vector_addition<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
    // Copy device variables to host
    cudaMemcpy(c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i=0; i<n; i++){
    //     printf("%d + %d = %d\n", a[i], b[i], c[i]);
    // }
    // Free memory on host and device
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}