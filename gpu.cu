#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA Kernel to perform long operation on the GPU
__global__ void long_operation_kernel(int* arr, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        arr[index] = index; // each thread updates its corresponding index
    }
}

int main() {
    // Define the size of the array
    const int size = 1e6;
    
    // Allocate host memory
    std::vector<int> arr(size);

    // Allocate device memory
    int* d_arr;
    cudaMalloc(&d_arr, size * sizeof(int));

    // Set up the grid and block dimensions
    int blockSize = 256; // Number of threads per block
    int numBlocks = (size + blockSize - 1) / blockSize; // Calculate number of blocks needed

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start time
    cudaEventRecord(start);

    // Launch the kernel
    long_operation_kernel<<<numBlocks, blockSize>>>(d_arr, size);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // Record the stop time
    cudaEventRecord(stop);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time: " << milliseconds << " milliseconds." << std::endl;

    // Copy the result back to host
    cudaMemcpy(arr.data(), d_arr, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_arr);

    // Optionally print some results to verify (e.g., first few elements)
    for (int i = 0; i < 10; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
