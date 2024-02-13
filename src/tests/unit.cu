#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "basic.cuh"


extern "C" void run_compute_mean(double *data, double *mean, int size, int threads_i) {
    double *d_data, *d_mean;
    cudaMalloc(&d_data, size * sizeof(double));
    cudaMalloc(&d_mean, sizeof(double));

    cudaMemcpy(d_data, data, size * sizeof(double), cudaMemcpyHostToDevice);

    int threads = threads_i;
    int blocks = (size + threads - 1) / threads;
    int sharedSize = threads * sizeof(double);

    compute_mean<<<blocks, threads, sharedSize>>>(d_data, d_mean, size);

    cudaMemcpy(mean, d_mean, sizeof(double), cudaMemcpyDeviceToHost);
    *mean /= size;

    cudaFree(d_data);
    cudaFree(d_mean);
}


extern "C" void run_compute_covariance_variance(double *xi, double *xj, double *mean_xi, double *mean_xj, double *covariance, double *variance, int size, int blockSize) {
    double *d_xi, *d_xj, *d_mean_xi, *d_mean_xj, *d_covariance, *d_variance;

    // Allocate and copy memory to device
    cudaMalloc(&d_xi, size * sizeof(double));
    cudaMalloc(&d_xj, size * sizeof(double));
    cudaMalloc(&d_mean_xi, sizeof(double));
    cudaMalloc(&d_mean_xj, sizeof(double));
    cudaMalloc(&d_covariance, sizeof(double));
    cudaMalloc(&d_variance, sizeof(double));

    cudaMemcpy(d_xi, xi, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xj, xj, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean_xi, mean_xi, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean_xj, mean_xj, sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(blockSize);
    dim3 grid((size + block.x - 1) / block.x);
    size_t sharedSize = 2 * block.x * sizeof(double);

    compute_covariance_variance<<<grid, block, sharedSize>>>(d_xi, d_xj, d_mean_xi, d_mean_xj, d_covariance, d_variance, size);

    cudaMemcpy(covariance, d_covariance, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(variance, d_variance, sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_xi);
    cudaFree(d_xj);
    cudaFree(d_mean_xi);
    cudaFree(d_mean_xj);
    cudaFree(d_covariance);
    cudaFree(d_variance);
}


extern "C" void run_element_wise_division(double *r, double *constant_std, double *result, int n, int blockSize) {
    double *d_r, *d_constant_std, *d_result;

    // Allocate and copy memory to device
    cudaMalloc(&d_r, n * sizeof(double));
    cudaMalloc(&d_constant_std, sizeof(double));
    cudaMalloc(&d_result, n * sizeof(double));

    cudaMemcpy(d_r, r, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_constant_std, constant_std, sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(blockSize);
    dim3 grid((n + block.x - 1) / block.x);

    element_wise_division<<<grid, block>>>(d_r, d_constant_std, d_result, n);

    cudaMemcpy(result, d_result, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_r);
    cudaFree(d_constant_std);
    cudaFree(d_result);
}



extern "C" void run_compute_residual(double *xi, double *xj, double *scaling_factor, double *residual, int size, int blockSize) {
    double *d_xi, *d_xj, *d_scaling_factor, *d_residual;

    // Allocate and copy memory to device
    cudaMalloc(&d_xi, size * sizeof(double));
    cudaMalloc(&d_xj, size * sizeof(double));
    cudaMalloc(&d_scaling_factor, sizeof(double));
    cudaMalloc(&d_residual, size * sizeof(double));

    cudaMemcpy(d_xi, xi, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xj, xj, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scaling_factor, scaling_factor, sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(blockSize);
    dim3 grid((size + block.x - 1) / block.x);

    compute_residual<<<grid, block>>>(d_xi, d_xj, d_scaling_factor, d_residual, size);

    cudaMemcpy(residual, d_residual, size * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_xi);
    cudaFree(d_xj);
    cudaFree(d_scaling_factor);
    cudaFree(d_residual);
}





extern "C" void run_compute_std(double *A, double *mean, double *std, int size, int blockSize) {
    double *d_A, *d_mean, *d_std;

    // Allocate and copy memory to device
    cudaMalloc(&d_A, size * sizeof(double));
    cudaMalloc(&d_mean, sizeof(double));
    cudaMalloc(&d_std, sizeof(double));

    cudaMemcpy(d_A, A, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, mean, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_std, std, sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(blockSize);
    dim3 grid((size + block.x - 1) / block.x);
    size_t sharedSize = block.x * sizeof(double);

    compute_std<<<grid, block, sharedSize>>>(d_A, d_mean, d_std, size);

    cudaMemcpy(std, d_std, sizeof(double), cudaMemcpyDeviceToHost);
    *std /= size;
    *std = sqrtf(*std);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_mean);
    cudaFree(d_std);
}


extern "C" void run_calculate_statistics(double *A, int m, int n, double *means, double *stds, int blockSize) {
    double *d_A, *d_means, *d_stds;

    // Allocate and copy memory to device
    cudaMalloc(&d_A, m * n * sizeof(double));
    cudaMalloc(&d_means, n * sizeof(double));
    cudaMalloc(&d_stds, n * sizeof(double));

    cudaMemcpy(d_A, A, m * n * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(blockSize);
    dim3 grid((n + block.x - 1) / block.x);

    calculate_statistics<<<grid, block>>>(d_A, m, n, d_means, d_stds);

    cudaMemcpy(means, d_means, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(stds, d_stds, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_means);
    cudaFree(d_stds);
}



extern "C" void run_standardize_column(double *A, int m, int n, double *means, double *stds, int threads_i) {
    double *d_A;
    double *d_means, *d_stds;

    // Allocate and copy memory to device
    cudaMalloc(&d_A, m * n * sizeof(double));
    cudaMalloc(&d_means, n * sizeof(double));
    cudaMalloc(&d_stds, n * sizeof(double));

    cudaMemcpy(d_A, A, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, means, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stds, stds, n * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(threads_i, threads_i);
    int numChunks = (m + threads_i - 1) / threads_i;

    for (int chunk = 0; chunk < numChunks; ++chunk) {
        int startRow = chunk *  threads_i;
        dim3 grid((threads_i + block.x - 1) / block.x,
                         (n + block.y - 1) / block.y);
        standardize_column<<<grid, block>>>(d_A, m, n, d_means, d_stds, startRow);
    }

    cudaMemcpy(A, d_A, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_means);
    cudaFree(d_stds);
}



extern "C" void run_compute_log_cosh(double *u, double *log_cosh_sum, int size, int blockSize) {
    double *d_u, *d_log_cosh_sum;

    cudaMalloc(&d_u, size * sizeof(double));
    cudaMalloc(&d_log_cosh_sum, sizeof(double));

    cudaMemcpy(d_u, u, size * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(blockSize);
    dim3 grid((size + block.x - 1) / block.x);
    size_t sharedSize = block.x * sizeof(double);

    compute_log_cosh<<<grid, block, sharedSize>>>(d_u, d_log_cosh_sum, size);

    cudaMemcpy(log_cosh_sum, d_log_cosh_sum, sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_u);
    cudaFree(d_log_cosh_sum);
}



extern "C" void run_compute_u_exp(double *u, double *u_exp_sum, int size, int blockSize) {
    double *d_u, *d_u_exp_sum;

    cudaMalloc(&d_u, size * sizeof(double));
    cudaMalloc(&d_u_exp_sum, sizeof(double));

    cudaMemcpy(d_u, u, size * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(blockSize);
    dim3 grid((size + block.x - 1) / block.x);
    size_t sharedSize = block.x * sizeof(double);

    compute_u_exp<<<grid, block, sharedSize>>>(d_u, d_u_exp_sum, size);

    cudaMemcpy(u_exp_sum, d_u_exp_sum, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_u);
    cudaFree(d_u_exp_sum);
}



extern "C" void run_process_column(double *X, double *column, int m, int n, int col_idx, int threads_i) {
    double *d_column;
    double *d_X;

    cudaMalloc(&d_X, m * n * sizeof(double));
    cudaMemcpy(d_X, X, m * n  * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_column, m * sizeof(double));

    int blocks = (m + threads_i - 1) / threads_i;
    process_column<<<blocks, threads_i>>>(d_X, d_column, m, n, col_idx);

    cudaMemcpy(column, d_column, m * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_column);
}



extern "C" void end_end_residual(double *data, int M, int N, int m, int *U, int uN)
{
    double *d_X;

    cudaMalloc(&d_X, M * N * sizeof(double));
    cudaMemcpy(d_X, data, M * N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blocks(256);
    dim3 grid((M + blocks.x - 1) / blocks.x);

    size_t sharedMemSize = blocks.x * sizeof(double);

    double *xj, *xm;

    cudaMalloc(&xj, M * sizeof(double));
    cudaMalloc(&xm, M * sizeof(double));

    double *means_xj, *means_xm;

    cudaMalloc(&means_xj, sizeof(double));
    cudaMalloc(&means_xm, sizeof(double));

    double *h_means_xj, *h_means_xm;

    cudaMallocHost(&h_means_xj, sizeof(double));
    cudaMallocHost(&h_means_xm, sizeof(double));

    double *d_residual_ij, *d_covariance_i, *d_variance_j, *scaling_factor_ij;

    cudaMalloc(&d_residual_ij, M * sizeof(double));

    cudaMalloc(&d_covariance_i, sizeof(double));
    cudaMalloc(&d_variance_j, sizeof(double));
    cudaMalloc(&scaling_factor_ij, sizeof(double));

    int j = 0;
    for (int uj = 0; uj < uN; ++uj)
    {
        j = U[uj];
        if (j != m)
        {
            process_column<<<grid, blocks>>>(d_X, xj, M, N, j);
            process_column<<<grid, blocks>>>(d_X, xm, M, N, m);

            compute_mean<<<grid, blocks, sharedMemSize>>>(xj, means_xj, M);
            compute_mean<<<grid, blocks, sharedMemSize>>>(xm, means_xm, M);

            divonhost(h_means_xj, means_xj, M);
            divonhost(h_means_xm, means_xm, M);

            compute_covariance_variance<<<grid, blocks, 2 * sharedMemSize>>>(xj, xm, means_xj, means_xm, d_covariance_i, d_variance_j, M);
            element_wise_division<<<grid, blocks>>>(d_covariance_i, d_variance_j, scaling_factor_ij, 1);

            compute_residual<<<grid, blocks>>>(xj, xm, scaling_factor_ij, d_residual_ij, M);
            update(d_X, d_residual_ij, M, N, j);

        }
    }

    cudaMemcpy(data, d_X, M * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(xj);
    cudaFree(xm);
    cudaFree(means_xj);
    cudaFree(means_xm);
    cudaFree(d_residual_ij);
    cudaFree(d_covariance_i);
    cudaFree(d_variance_j);
    cudaFree(scaling_factor_ij);
    cudaFreeHost(h_means_xj);
    cudaFreeHost(h_means_xm);
    cudaFree(d_X);

}
