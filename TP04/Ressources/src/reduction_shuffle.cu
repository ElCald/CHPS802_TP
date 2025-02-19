////////////////////////////////////////////////////////////////////////
//
// Practical 4 -- initial code for shared memory reduction for 
//                a single block which is a power of two in size
//
////////////////////////////////////////////////////////////////////////

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


__global__ void reduction(float *g_odata, float *g_idata, int num_elements)
{

    __shared__ float partial_sums[32];  

    int tid = threadIdx.x; 
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    float sum = 0.0f;

    
    if( idx < num_elements ){
        sum = g_idata[idx];
    }


    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }


    if (tid % warpSize == 0) {  
        partial_sums[tid / warpSize] = sum;
    }
    
    __syncthreads();


    if (tid < warpSize) {
        sum = (tid < blockDim.x / warpSize) ? partial_sums[tid] : 0.0f;

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        if (tid == 0) {
            g_odata[blockIdx.x] = sum;  
        }
    }
}


////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
    int num_blocks, num_threads, num_elements, mem_size, shared_mem_size;

    float *h_data, *d_idata, *d_odata, *h_odata, res_sum = 0.0f;



    // initialise card
    findCudaDevice(argc, argv);


    num_blocks   = 4;  // start with only 1 thread block
    num_threads  = 128;
    num_elements = num_blocks*num_threads;
    mem_size     = sizeof(float) * num_elements;



    // allocate host memory to store the input data
    h_data = (float*) malloc(mem_size);
    h_odata = (float*) malloc(sizeof(float) * num_blocks);

    // and initialize to integer values between 0 and 10
    for(int i = 0; i < num_elements; i++){
        h_data[i] = floorf(10.0f*(rand()/(float)RAND_MAX));
    }
        

    // compute reference solution
    float sum = reduction_gold(h_data, num_elements);


    // allocate device memory input and output arrays
    cudaMalloc((void**)&d_idata, mem_size);
    cudaMalloc((void**)&d_odata, num_blocks*sizeof(float));


    // copy host memory to device input array
    cudaMemcpy(d_idata, h_data, mem_size, cudaMemcpyHostToDevice) ;


    // execute the kernel
    shared_mem_size = sizeof(float) * num_threads;
    reduction<<<num_blocks, num_threads, shared_mem_size>>>(d_odata, d_idata, num_elements);
    getLastCudaError("reduction kernel execution failed");


    // copy device memory to host     
    cudaMemcpy(h_odata, d_odata, sizeof(float) * num_blocks, cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_blocks; i++) {
        res_sum += h_odata[i];
    }


    // check results
    printf("reduction error = %f\n", res_sum-sum);


    // cleanup memory
    free(h_data);
    cudaFree(d_idata);
    cudaFree(d_odata);


    // CUDA exit -- needed to flush printf write buffer
    cudaDeviceReset();
}
