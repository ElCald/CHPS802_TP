#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void my_first_kernel(float *tab_A, float *tab_B, float *res) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    res[tid] = tab_A[tid] + tab_B[tid] +1; // "+1" qui fausse la somme
}

TEST(CudaTest, SommeVecteur) {
    const int tab_size = 32;
    float *h_tab_A, *h_tab_B, *h_res;
    float *d_tab_A, *d_tab_B, *d_res;

    h_tab_A = (float*)malloc(tab_size * sizeof(float));
    h_tab_B = (float*)malloc(tab_size * sizeof(float));
    h_res = (float*)malloc(tab_size * sizeof(float));

    cudaMalloc((void**)&d_tab_A, tab_size * sizeof(float));
    cudaMalloc((void**)&d_tab_B, tab_size * sizeof(float));
    cudaMalloc((void**)&d_res, tab_size * sizeof(float));

    for (int i = 0; i < tab_size; i++) {
        h_tab_A[i] = i + 1;
        h_tab_B[i] = i + 5;
    }

    cudaMemcpy(d_tab_A, h_tab_A, tab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tab_B, h_tab_B, tab_size * sizeof(float), cudaMemcpyHostToDevice);

    my_first_kernel<<<4, 8>>>(d_tab_A, d_tab_B, d_res);
    cudaDeviceSynchronize();

    cudaMemcpy(h_res, d_res, tab_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < tab_size; i++) {
        //EXPECT_EQ(h_res[i], h_tab_A[i] + h_tab_B[i]);
        ASSERT_EQ(h_res[i], h_tab_A[i] + h_tab_B[i]);
    }

    cudaFree(d_tab_A);
    cudaFree(d_tab_B);
    cudaFree(d_res);
    free(h_tab_A);
    free(h_tab_B);
    free(h_res);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}