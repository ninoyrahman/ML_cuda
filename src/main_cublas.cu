// main.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include <math.h>
#include "util.h"
#include "kernel.h"
#include "cublas_v2.h"
 
// main routine that executes on the host
int main(void){

  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////declaration///////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  // varify gpu calculation
  bool isVarify = true;

  // profile variable
  float et;
  cudaEvent_t start, stop;

  // cublas handle
  cublasStatus_t status;
  cublasHandle_t handle;
  cublasCreate(&handle);

  // scaling factor
  float alpha;
  float beta;

  // matrix and vector size
  const int N1 = 4096; // matC row/matA row number
  const int N2 = 2048; // matC column/matB column number
  const int N3 = 1024; // matB row/matA column number

  // allocate on host
  float *matA_h  = new float[N1 * N3];
  float *matB_h  = new float[N3 * N2];
  float *matC_h  = new float[N1 * N2];
  float *vecB_h  = new float[N1];
  float *vecO_h  = new float[N2];
  
  // Pointer to device arrays
  float *matA_d, *matB_d, *matC_d;
  float *vecB_d, *vecO_d;

  // size of matrix
  size_t size_matA = N1 * N3 * sizeof(float);
  size_t size_matB = N3 * N2 * sizeof(float);
  size_t size_matC = N1 * N2 * sizeof(float);

  // size of vector
  size_t size_vecB = N1 * sizeof(float);
  size_t size_vecO = N2 * sizeof(float);

  // allocate on device
  cudaMalloc((void **) &matA_d, size_matA);
  cudaMalloc((void **) &matB_d, size_matB);
  cudaMalloc((void **) &matC_d, size_matC);
  cudaMalloc((void **) &vecB_d, size_vecB);
  cudaMalloc((void **) &vecO_d, size_vecO);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////initialization////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////

  // initialize performance profile variables
  cudaEventCreate(&start); cudaEventCreate(&stop);

  // print matrix size
  std::cout << " A matrix dims = (" << N1 << ", " << N3 << ")" << std::endl;
  std::cout << " B matrix dims = (" << N3 << ", " << N2 << ")" << std::endl;
  std::cout << " C matrix dims = (" << N1 << ", " << N2 << ")" << std::endl;
  std::cout << " B vector dims = (" << N1 << ")" << std::endl;
  std::cout << " O vector dims = (" << N2 << ")" << std::endl;

  // Initialize host array and copy it to CUDA device
  random_matrix(matA_h, N1, N3);
  random_matrix(matB_h, N3, N2);
  random_vector(vecB_h, N1);
  setvalue_vector(vecO_h, N2, 1.0f);
  
  // copy data from host to device
  cudaEventRecord(start);
  cudaMemcpy(matA_d, matA_h, size_matA, cudaMemcpyHostToDevice); // copy matA to device
  cudaMemcpy(matB_d, matB_h, size_matB, cudaMemcpyHostToDevice); // copy matB to device
  cudaMemcpy(vecB_d, vecB_h, size_vecB, cudaMemcpyHostToDevice); // copy vecB to device
  cudaMemcpy(vecO_d, vecO_h, size_vecO, cudaMemcpyHostToDevice); // copy vecO to device
  cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&et, start, stop);
  std::cout << "time elapsed(copy to device) = " << et << std::endl;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////computation///////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////  
  
  // gpu matrix multiplication
  cudaEventRecord(start);
  alpha = 1.0f;
  beta = 0.0f;
  status = cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N1, N2, N3,
    &alpha,
    matA_d, N1,
    matB_d, N3,
    &beta,
    matC_d, N1);
  cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&et, start, stop);
  std::cout << "time elapsed(mat x mat) = " << et << std::endl;

  if (status != CUBLAS_STATUS_SUCCESS) 
    printf("cublasSgemm returned error code %d\n", status);

  if (isVarify){
    // Retrieve result from device and store it in host array
    cudaEventRecord(start);
    cudaMemcpy(matC_h, matC_d, size_matC, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&et, start, stop);
    std::cout << "time elapsed(copy from device) = " << et << std::endl;

    // cpu matrix multiplication 
    // copy matC from device to host before calling varify routine
    mat_mul_varify(matA_h, matB_h, matC_h, N1, N2, N3);
  }

  // gpu matrix vector addition
  cudaEventRecord(start);
  alpha = 1.0f;
  status = cublasSger(handle, 
    N1, N2,
    &alpha,
    vecB_d, 1,
    vecO_d, 1,
    matC_d, N1);
  cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&et, start, stop);
  std::cout << "time elapsed(mat + vec) = " << et << std::endl;

  if (status != CUBLAS_STATUS_SUCCESS) 
    printf("cublasSger returned error code %d\n", status);

  // Retrieve result from device and store it in host array
  cudaEventRecord(start);
  cudaMemcpy(matC_h, matC_d, size_matC, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&et, start, stop);
  std::cout << "time elapsed(copy from device) = " << et << std::endl;    

  if (isVarify){
    // cpu matrix multiplication and vector addition 
    // copy matC from device to host before calling varify routine
    mat_mul_vec_sum_varify(matA_h, matB_h, matC_h, vecB_h, N1, N2, N3);
  }
  
  // Print results
  // print_matrix(matC_h, N1, N2);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////finalization//////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////    
  
  // Cleanup
  delete [] matA_h;
  delete [] matB_h;
  delete [] matC_h;
  delete [] vecB_h;
  delete [] vecO_h;
  cudaFree(matA_d);
  cudaFree(matB_d);
  cudaFree(matC_d);
  cudaFree(vecB_d);
  cudaFree(vecO_d);

  return 0;
}