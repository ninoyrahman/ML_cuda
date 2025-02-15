// main.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include <math.h>
#include "util.h"
#include "kernel.h"
#include "cublas_v2.h"
 
// main routine that executes on the host
int main(void){

  // cublas handle
  cublasStatus_t status;
  cublasHandle_t handle;
  cublasCreate(&handle);

  // scaling factor
  float alpha = 1.0f;
  float beta  = 0.0f;  

  float *matA_h, *matB_h, *matC_h;  // Pointer to host & device arrays
  float *matA_d, *matB_d, *matC_d;  // Pointer to host & device arrays
  const int N1 = 4096; // matC row/matA row number
  const int N2 = 2048; // matC column/matB column number
  const int N3 = 1024;    // matB row/matA column number
  const int nthreads = 32; // number of threads

  size_t size_matA = N1 * N3 * sizeof(float);
  size_t size_matB = N3 * N2 * sizeof(float);
  size_t size_matC = N1 * N2 * sizeof(float);

  matA_h  = new float[N1 * N3];    // Allocate array on host
  matB_h  = new float[N3 * N2];    // Allocate array on host
  matC_h  = new float[N1 * N2];    // Allocate array on host

  float *matCr_h  = new float[N1 * N2];    // Allocate array on host
  
  cudaMalloc((void **) &matA_d, size_matA);   // Allocate array on device
  cudaMalloc((void **) &matB_d, size_matB);   // Allocate array on device
  cudaMalloc((void **) &matC_d, size_matC);   // Allocate array on device

  // Initialize host array and copy it to CUDA device
  random_matrix(matA_h, N1, N3);
  random_matrix(matB_h, N3, N2); 
  
  // copy data from host to device
  float et;
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaMemcpy(matA_d, matA_h, size_matA, cudaMemcpyHostToDevice); // copy matA to device
  cudaMemcpy(matB_d, matB_h, size_matB, cudaMemcpyHostToDevice); // copy matB to device
  cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&et, start, stop);
  std::cout << "time elapsed(copy to device) = " << et << std::endl;

  // Do calculation on device
  dim3 threadsPerBlock(nthreads, nthreads);
  dim3 blocksPerGrid((int)ceil(N1 / threadsPerBlock.x), (int)ceil(N2 / threadsPerBlock.y));

  std::cout << " A matrix dims = (" << N1 << ", " << N3 << ")" << std::endl;
  std::cout << " B matrix dims = (" << N3 << ", " << N2 << ")" << std::endl;
  std::cout << " C matrix dims = (" << N1 << ", " << N2 << ")" << std::endl;     
  std::cout << " thread per block = (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << std::endl;
  std::cout << " block number = (" << blocksPerGrid.x << ", " << blocksPerGrid.y << ")" << std::endl;
  
  // gpu matrix multiplication
  cudaEventRecord(start);
  status = cublasSgemm(handle,
    CUBLAS_OP_T, CUBLAS_OP_T,
    N1, N2, N3,
    &alpha,
    matA_d, N3,
    matB_d, N2,
    &beta,
    matC_d, N1);
  cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&et, start, stop);
  std::cout << "time elapsed(kernel) = " << et << std::endl;

  if (status != CUBLAS_STATUS_SUCCESS) 
    printf("cublasSgemm returned error code %d\n", status);

  // Retrieve result from device and store it in host array
  cudaEventRecord(start);
  cudaMemcpy(matCr_h, matC_d, size_matC, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&et, start, stop);
  std::cout << "time elapsed(copy from device) = " << et << std::endl;

  // cpu matrix multiplication
  rearrange_matrix(matCr_h, matC_h, N1, N2);
  mat_mul_varify(matA_h, matB_h, matC_h, N1, N2, N3);
  
  // Print results
  // print_matrix(matC_h, N1, N2);
  
  // Cleanup
  delete [] matA_h;
  delete [] matB_h;
  delete [] matC_h;
  delete [] matCr_h;
  cudaFree(matA_d);
  cudaFree(matB_d);
  cudaFree(matC_d);

  return 0;
}