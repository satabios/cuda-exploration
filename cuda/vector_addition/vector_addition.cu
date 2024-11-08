#include<stdio.h>

__global__ void vector_addition(int *a, int *b, int *c, int n){
    // printf("ThreadIdx: %d, BlockIdx: %d blockDim: %d\n", threadIdx.x, blockIdx.x, blockDim.x);
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n){
        c[i] = a[i] + b[i];
    }
}



int main(){
    int n = 1<<4;

    int *a, *b, *c; // host variables
    int *d_a, *d_b, *d_c; // device variables
    // Memory allocation on host
    a = (int*)malloc(n*sizeof(int)); 
    b = (int*)malloc(n*sizeof(int));
    c = (int*)malloc(n*sizeof(int));
    // Memory allocation on device
    cudaMalloc(&d_a, n*sizeof(int));
    cudaMalloc(&d_b, n*sizeof(int));
    cudaMalloc(&d_c, n*sizeof(int));
    // Initialize host variables
    for(int i=0; i<n; i++){
        a[i] = i;
        b[i] = i;
    }
    // Copy host variables to device
    cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice);
    // Kernel launch
    dim3 blockDim(16);
    dim3 gridDim((n+blockDim.x-1)/blockDim.x); // (16+16-1)/16 = 2 ; General Formula: (n+blockDim-1)/blockDim for number that are not multiple of blockDim
    vector_addition<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);
    // Copy device variables to host
    cudaMemcpy(c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0; i<n; i++){
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    // Free memory on host and device
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}