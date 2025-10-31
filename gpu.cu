#include <iostream>
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
    const int size = 1e6;  // Set the array size (1 million elements)

    // Allocate host memory
    int* arr = new int[size];  

    // Declare device memory pointer
    int* d_arr;

    // Create CUDA events for measuring time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start time (for the entire operation)
    cudaEventRecord(start);

    // Step 1: Allocate memory on the GPU
    cudaMalloc(&d_arr, size * sizeof(int));

    // Step 2: Set up the grid and block dimensions
    int blockSize = 256; // Number of threads per block
    int numBlocks = (size + blockSize - 1) / blockSize; // Calculate number of blocks needed

    // Step 3: Launch the kernel
    long_operation_kernel<<<numBlocks, blockSize>>>(d_arr, size);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // Step 4: Copy the result back to host
    cudaMemcpy(arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Step 5: Free device memory
    cudaFree(d_arr);

    // Record the stop time after all operations are done
    cudaEventRecord(stop);

    // Synchronize events to make sure everything has completed
    cudaEventSynchronize(stop);

    // Calculate the elapsed time for the entire operation
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Total operation time: " << milliseconds << " milliseconds." << std::endl;

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Optionally print some results to verify
    std::cout << "First 10 values: ";
    for (int i = 0; i < 10; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    // Clean up the dynamically allocated array
    delete[] arr;

    return 0;
}