////////////////////////////////////////////////////////////////////////
//
// Practical 4 -- initial code for shared memory reduction for 
//                a single block which is a power of two in size
//
////////////////////////////////////////////////////////////////////////


#include <gtest/gtest.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// CPU routine
////////////////////////////////////////////////////////////////////////

float reduction_gold(float* idata, int len) 
{
  float sum = 0.0f;
  for(int i=0; i<len; i++) sum += idata[i];

  return sum;
}

////////////////////////////////////////////////////////////////////////
// GPU routine
////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *g_odata, float *g_idata)
{
    // dynamically allocated shared memory

    extern  __shared__  float temp[];

    int tid = threadIdx.x;

    // first, each thread loads data into shared memory

    temp[tid] = g_idata[tid];

    // next, we perform binary tree reduction

    for (int d=blockDim.x/2; d>0; d=d/2) {
        __syncthreads();  // ensure previous step completed 
        if (tid<d)  temp[tid] += temp[tid+d];
    }

    // finally, first thread puts result into global memory

    if (tid==0) g_odata[0] = temp[0];
}


////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////

int g_argc;
const char** g_argv;

TEST(CudaTest, Reduction) 
{
    int num_blocks, num_threads, num_elements, mem_size, shared_mem_size;

    float *h_data, *d_idata, *d_odata;


    // initialise card
    findCudaDevice(g_argc, g_argv);


    num_blocks   = 1;  // start with only 1 thread block
    num_threads  = 512;
    num_elements = num_blocks*num_threads;
    mem_size     = sizeof(float) * num_elements;


    // allocate host memory to store the input data
    h_data = (float*) malloc(mem_size);
    h_data = (float*) malloc(mem_size);

    // and initialize to integer values between 0 and 10
    for(int i = 0; i < num_elements; i++){
        h_data[i] = floorf(10.0f*(rand()/(float)RAND_MAX));
    }
        

    // compute reference solution
    float sum = reduction_gold(h_data, num_elements);


    // allocate device memory input and output arrays
    cudaMalloc((void**)&d_idata, mem_size);
    cudaMalloc((void**)&d_odata, sizeof(float));


    // copy host memory to device input array
    cudaMemcpy(d_idata, h_data, mem_size, cudaMemcpyHostToDevice) ;


    // execute the kernel
    shared_mem_size = sizeof(float) * num_threads;
    reduction<<<num_blocks, num_threads, shared_mem_size>>>(d_odata, d_idata);
    getLastCudaError("reduction kernel execution failed");


    // copy result from device to host
    cudaMemcpy(h_data, d_odata, sizeof(float), cudaMemcpyDeviceToHost);


    ASSERT_FLOAT_EQ(sum, h_data[0]);

    // check results
    printf("reduction error = %f\n",h_data[0]-sum);


    // cleanup memory
    free(h_data);
    cudaFree(d_idata);
    cudaFree(d_odata);


    // CUDA exit -- needed to flush printf write buffer
    cudaDeviceReset();
}


int main(int argc, char **argv) {
    
    g_argc = argc;
    g_argv = const_cast<const char**>(argv);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}