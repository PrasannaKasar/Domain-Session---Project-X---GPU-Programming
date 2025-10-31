#include <iostream>
#include <vector>
#include <ctime>

void matrix_multiply(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& C) {
    int n = A.size();
    int m = B[0].size();
    int k = B.size();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            C[i][j] = 0;
            for (int l = 0; l < k; ++l) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

int main() {
    int n = 1, k = 1, m = 1; // For example, matrices of size 1024x1024

    // Initialize matrices A and B with random values
    std::vector<std::vector<int>> A(n, std::vector<int>(k, 1));
    std::vector<std::vector<int>> B(k, std::vector<int>(m, 1));
    std::vector<std::vector<int>> C(n, std::vector<int>(m, 0));

    clock_t start_time = clock();
    
    // Perform matrix multiplication
    matrix_multiply(A, B, C);

    clock_t end_time = clock();
    double duration = double(end_time - start_time) / CLOCKS_PER_SEC * 1000; // milliseconds
    std::cout << "CPU matrix multiplication time: " << duration << " ms" << std::endl;

    return 0;
}