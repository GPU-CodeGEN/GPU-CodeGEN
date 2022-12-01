#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "kernel.h"
#include <stdlib.h>

#define TILE_DIM 16

using namespace std;

__global__ void matrixMultiplicationKernel(int m, int k, int n, float* A, float* B, float* C) {
    __shared__ float ATile[TILE_DIM][TILE_DIM];
    __shared__ float BTile[TILE_DIM][TILE_DIM];

    // int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float Cvalue = 0;

    // Loop over the A and B tiles required to compute the C element
    for (int t = 0; t < (k-1)/TILE_DIM +1; ++t)
    {
        // Collaborative loading of A and B tiles into shared memory
        if(row < m && t*TILE_DIM+tx < k)
            ATile[ty][tx] = A[row*k + t*TILE_DIM+tx];
        else
            ATile[ty][tx] = 0.0f;
        
        if (t*TILE_DIM+ty < k && col < n)
            BTile[ty][tx] = B[(t*TILE_DIM+ty)*n + col];
        else
            BTile[ty][tx] = 0.0f;
        
        __syncthreads();

        for (int i = 0; i < TILE_DIM; ++i)
            Cvalue += ATile[ty][i] * BTile[i][tx];

        __syncthreads();
    }
    if (row < m && col < n)
        C[row*n+col] = Cvalue;
}

void matrixMultiplication(int m, int k, int n, float *A, float *B, float *C){

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(m, k, n, A, B, C);
}