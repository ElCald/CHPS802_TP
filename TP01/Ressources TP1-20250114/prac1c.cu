%%cuda
//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "/content/drive/MyDrive/Cours./s8/CHPS802 - GPU/TP/TP01/Fichiers dentête-20250114/helper_cuda.h"


//
// kernel routine
//

__global__ void my_first_kernel(float *tab_A, float *tab_B, float *res)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    res[tid] = tab_A[tid] + tab_B[tid];

    printf("%d > %lf | %lf\n", threadIdx.x, tab_A[tid], tab_B[tid]);
}


//
// main code
//

int main(int argc, const char **argv)
{

  int nblocks, nthreads, nsize, n;
  float *h_tab_A, *h_tab_B, *h_res, *h_res_test;
  float *d_tab_A, *d_tab_B, *d_res;
  const int tab_size = 32;

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks  = 4;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory for array

  h_tab_A = (float*)malloc(tab_size*sizeof(float));
  h_tab_B = (float*)malloc(tab_size*sizeof(float));
  h_res = (float*)malloc(tab_size*sizeof(float));
  h_res_test = (float*)malloc(tab_size*sizeof(float));

  cudaMalloc((void**)&d_tab_A, tab_size * sizeof(float));
  cudaMalloc((void**)&d_tab_B, tab_size * sizeof(float));
  cudaMalloc((void**)&d_res, tab_size * sizeof(float));


  // init tab

  for(int i=0; i<tab_size; i++){
      h_tab_A[i] = i+1;
      h_tab_B[i] = i+5;
  }


  // copy data host 2 device

  cudaMemcpy(d_tab_A, h_tab_A, tab_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tab_B, h_tab_B, tab_size*sizeof(float), cudaMemcpyHostToDevice);


  // execute kernel

  my_first_kernel<<<nblocks,nthreads>>>(d_tab_A, d_tab_B, d_res);
  getLastCudaError("my_first_kernel execution failed\n");


  // copy data device 2 host

  cudaMemcpy(h_res, d_res, tab_size*sizeof(float), cudaMemcpyDeviceToHost);


  // Test host & device

  for(int i=0; i<tab_size; i++){
      h_res_test[i] = h_tab_A[i] + h_tab_B[i];
  }

  // print data

  for(int i=0; i<tab_size; i++){
      printf("%lf ", h_res[i]);
  }
  printf("\n");

  for(int i=0; i<tab_size; i++){
      printf("%lf ", h_res_test[i]);
  }
  printf("\n");


  // synchornize host & device

  cudaDeviceSynchronize();


  // free memory

  checkCudaErrors(cudaFree(d_tab_A));
  checkCudaErrors(cudaFree(d_tab_B));
  checkCudaErrors(cudaFree(d_res));
  free(h_tab_A);
  free(h_tab_B);
  free(h_res);


  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
