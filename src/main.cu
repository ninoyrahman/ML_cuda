// main.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include "util.h"
 
// main routine that executes on the host
int main(void)
{

  float *a_h, *b_h, *c_h;  // Pointer to host & device arrays
  float *a_d, *b_d, *c_d;  // Pointer to host & device arrays
  const int N = 1024;  // Number of elements in arrays
  size_t size = N * sizeof(float);

  a_h = new float[size];    // Allocate array on host
  b_h = new float[size];    // Allocate array on host
  c_h = new float[size];    // Allocate array on host  
  cudaMalloc((void **) &a_d, size);   // Allocate array on device
  cudaMalloc((void **) &b_d, size);   // Allocate array on device
  cudaMalloc((void **) &c_d, size);   // Allocate array on device

  // Initialize host array and copy it to CUDA device
  std::srand(std::time({}));
  for (int i=0; i<N; i++){ 
    a_h[i] = (float(rand()) / RAND_MAX) - 0.5;
    b_h[i] = (float(rand()) / RAND_MAX) - 0.5;
  }

  // copy data from host to device
  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
  
  // Do calculation on device:
  int block_size = 512;
  int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);

  dim3 threadsPerBlock(block_size);
  dim3 blocksPerGrid(n_blocks);

  std::cout << " thread per block = " << block_size << std::endl;
  std::cout << " block number = " << n_blocks << std::endl;

  // square array
  // square_array <<< n_blocks, block_size >>> (a_d, b_d, c_d, N);
  square_array <<< blocksPerGrid, threadsPerBlock >>> (a_d, b_d, c_d, N);

  // Retrieve result from device and store it in host array
  cudaMemcpy(c_h, c_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
  
  // Print results
  // for (int i=0; i<N; i++){
  //   printf("%d %f %f %f\n", i, a_h[i], b_h[i], c_h[i]);
  // }
  
  // Cleanup
  delete [] a_h;
  delete [] b_h;
  delete [] c_h;
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  return 0;
}