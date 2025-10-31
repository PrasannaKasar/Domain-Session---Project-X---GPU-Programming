#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, int n, int k, int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < n && col < m) {
        float value = 0.0f;
        for (int i = 0; i < k; ++i) {
            value += A[row * k + i] * B[i * m + col]; // Matrix multiplication formula
        }
        C[row * m + col] = value;
    }
}

int main() {

    // Record the start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int n = 100, k = 100, m = 100;  // For example, matrices of size n x k and k x m

    // Initialize matrices A and B with random values
    std::vector<float> A(n * k, 1.0f); // Matrix A (n x k)
    std::vector<float> B(k * m, 1.0f); // Matrix B (k x m)
    std::vector<float> C(n * m, 0.0f); // Matrix C (n x m) to store the result

    // Allocate memory for device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * m * sizeof(float));
    cudaMalloc((void**)&d_C, n * m * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A.data(), n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), k * m * sizeof(float), cudaMemcpyHostToDevice);

    // Set up the grid and block dimensions
    dim3 blockSize(16, 16); // Each block will have 16x16 threads
    dim3 numBlocks((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrix_multiply_kernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n, k, m);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Record the stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU matrix multiplication time: " << milliseconds << " ms" << std::endl;

    // Copy the result matrix C back to host
    cudaMemcpy(C.data(), d_C, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}