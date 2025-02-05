#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_cuda.h>



////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ float  a=20.0f, b=25.0f, c=15.0f;



////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////

__global__ void kernel_gen_rand_nums(unsigned long seed, float *d_tab, int nb_val)
{
    curandState state;
    curand_init(seed, threadIdx.x, 0, &state);

    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    double z, som=0;


    for(int i =0; i<nb_val; i++){
      z = curand_normal(&state);  // Génère un nombre aléatoire entre 0 et 1
      som += (a*pow(z,2.0) + b * z + c);
    }

    d_tab[tid] = som/nb_val;

}


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){

  // variables
  float *d_tab, *h_tab; // tableau contenant la valeur générée par chaque thread
  int d_size_tab; // Taille du tableau
  int nb_val;             // Nombre de valeurs générées par thread
  int nb_blocks, nb_threads; // Nombre de blocs et nombre de threads
  float som_total=0, moyenne=0;
  const float d_a=20.0f, d_c=15.0f;

  curandState *devStates; // Etat du générateur


  // init size blocks / threads / tab
  nb_blocks = 128;
  nb_threads = 128;
  d_size_tab = nb_blocks * nb_threads;
  nb_val = 200;


  // Allocation
  cudaMalloc((void**)&devStates, nb_threads * sizeof(curandState));
  checkCudaErrors(cudaMallocManaged(&d_tab, d_size_tab*sizeof(float)));
  h_tab = (float*)malloc(sizeof(float) * d_size_tab);



  // execute kernel

  kernel_gen_rand_nums<<<nb_blocks, nb_threads>>>(time(NULL), d_tab, nb_val);

  cudaDeviceSynchronize();


  // Copy device 2 host
  checkCudaErrors( cudaMemcpy(h_tab, d_tab, sizeof(float)*d_size_tab, cudaMemcpyDeviceToHost) );


  // Calcul moyenne

  for(int i=0; i<d_size_tab; i++){
    som_total += h_tab[i];
  }

  moyenne = som_total / d_size_tab;

  printf("Moyenne: %lf\n", moyenne);
  printf("a + c: %lf\n", d_a+d_c);




  // Libération de la mémoire
  free(h_tab);
  checkCudaErrors( cudaFree(d_tab) );
  checkCudaErrors( cudaFree(devStates) );


  // CUDA exit -- needed to flush printf write buffer
  cudaDeviceReset();

}