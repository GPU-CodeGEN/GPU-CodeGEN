#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "kernel.cu"
#include "dev_array.h"
#include <random>
#include <math.h>

#define NUM_PROFILES 512
#define M 128
#define K 64
#define N 256

using namespace std;

template <typename ValueType>
void MakeDenseMatrix(int rows, int columns, ValueType *matrix,
                     std::default_random_engine generator)
{
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);
    
    for (int64_t i = 0; i < static_cast<int64_t>(rows) * columns; ++i){
        float temp = distribution(generator);
        matrix[i] = ValueType(temp);
        // int temp = (i / columns) % 8;
        // matrix[i] = half(temp * 0.01);
    }
}

int main()
{
    // Perform matrix multiplication C = A*B
    int m, k, n;
    m = M;
    k = K;
    n = N;
    std::default_random_engine generator;

    // Allocate memory on the host
    float* h_A = new float[m * k];
    float* h_B = new float[k * n];
    float* h_C = new float[m * n];

    MakeDenseMatrix<float>(m, k, h_A, generator);
    MakeDenseMatrix<float>(k, n, h_B, generator);

    // Allocate memory on the device
    dev_array<float> d_A(m * k);
    dev_array<float> d_B(k * n);
    dev_array<float> d_C(m * n);

    d_A.set(&h_A[0], m * k);
    d_B.set(&h_B[0], k * n);

    float matmul_ms_avg = 0.0f;
    for(int iter=0; iter<NUM_PROFILES; ++iter) {
        float matmul_ms = 0.0f;
        cudaEvent_t matmul_start;
        cudaEvent_t matmul_end;
        cudaEventCreate(&matmul_start);
        cudaEventCreate(&matmul_end);
        cudaEventRecord(matmul_start);
        matrixMultiplication(m, k, n, d_A.getData(), d_B.getData(), d_C.getData());
        cudaDeviceSynchronize();
        cudaEventRecord(matmul_end);
        cudaEventSynchronize(matmul_end);
        cudaEventElapsedTime(&matmul_ms, matmul_start, matmul_end);
        cudaEventDestroy(matmul_start);
        cudaEventDestroy(matmul_end);
        matmul_ms_avg += matmul_ms;
    }
    std::cout << "Average MatMul time: " << matmul_ms_avg / NUM_PROFILES << " ms" << std::endl;

    d_C.get(&h_C[0], m * n);
    cudaDeviceSynchronize();

    float *cpu_C;
    cpu_C = new float[m * n];

    // Now do the matrix multiplication on the CPU
    float sum;
    for (int i=0; i < m; i++){
        for (int j=0; j < n; j++){
            sum = 0.f;
            for (int p = 0; p < k; p++){
                sum += h_A[i * k + p] * h_B[p * n + j];
            }
            cpu_C[i * n + j] = sum;
        }
    }

    double err = 0;
    // Check the result and make sure it is correct
    for (int ROW=0; ROW < m; ROW++){
        for (int COL=0; COL < n; COL++){
            err += cpu_C[ROW * n + COL] - h_C[ROW * n + COL];
        }
    }

    // for (int ROW=0; ROW < m; ROW++){
    //     for (int COL=0; COL < n; COL++){
    //         cout << cpu_C[ROW * n + COL] << " ";
    //     }
    //     cout << endl;
    // }

    // cout << endl;

    // for (int ROW=0; ROW < m; ROW++){
    //     for (int COL=0; COL < n; COL++){
    //         cout << h_C[ROW * n + COL] << " ";
    //     }
    //     cout << endl;
    // }

    // cout << endl;

    cout << "Error: " << err << endl;

    return 0;
}