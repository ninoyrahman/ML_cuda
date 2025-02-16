// main.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include <math.h>
#include "cublas_v2.h"
#include "util.h"
#include "nn.h"

// main routine that executes on the host
int main(void){

  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////declaration and allocation///////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////

  // matrix and vector size
  const int Ns = 50; // matB row/matA column number
  const int N0 = 784;  // matC row/matA row number
  const int N1 = 256;   // matC column/matB column number
  const int N2 = 128;   // matB row/matA column number
  const int N3 = 10;   // matB row/matA column number

  // allocate on host

  // data X, Y
  float *matX_h  = new float[N0 * Ns];
  float *matY_h  = new float[N3 * Ns];

  // weights
  float *matw1_h  = new float[N1 * N0];
  float *matw2_h  = new float[N2 * N1];
  float *matw3_h  = new float[N3 * N2];
  
  // biases
  float *vecb1_h  = new float[N1];
  float *vecb2_h  = new float[N2];
  float *vecb3_h  = new float[N3];
  
  // Pointer to device arrays
  float *matX_d, *matY_d;
  float *matw1_d, *matw2_d, *matw3_d;
  float *vecb1_d, *vecb2_d, *vecb3_d;

  // size of matrix
  size_t size_matX  = N0 * Ns * sizeof(float);
  size_t size_matY  = N3 * Ns * sizeof(float);

  size_t size_matw1 = N1 * N0 * sizeof(float);
  size_t size_matw2 = N2 * N1 * sizeof(float);
  size_t size_matw3 = N3 * N2 * sizeof(float);

  // size of vector
  size_t size_vecb1 = N1 * sizeof(float);
  size_t size_vecb2 = N2 * sizeof(float);
  size_t size_vecb3 = N3 * sizeof(float);

  // allocate on device
  cudaMalloc((void **) &matX_d, size_matX);
  cudaMalloc((void **) &matY_d, size_matY);

  cudaMalloc((void **) &matw1_d, size_matw1);
  cudaMalloc((void **) &matw2_d, size_matw2);
  cudaMalloc((void **) &matw3_d, size_matw3);

  cudaMalloc((void **) &vecb1_d, size_vecb1);
  cudaMalloc((void **) &vecb2_d, size_vecb2);
  cudaMalloc((void **) &vecb3_d, size_vecb3);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////initialization////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////

  // print matrix size
  std::cout << " X matrix dims = (" << N0 << ", " << Ns << ")" << std::endl;
  std::cout << " Y matrix dims = (" << N0 << ", " << Ns << ")" << std::endl;

  std::cout << " w1 matrix dims = (" << N1 << ", " << N0 << ")" << std::endl;
  std::cout << " w2 matrix dims = (" << N2 << ", " << N1 << ")" << std::endl;
  std::cout << " w3 matrix dims = (" << N3 << ", " << N2 << ")" << std::endl;

  std::cout << " b1 vector dims = (" << N1 << ")" << std::endl;
  std::cout << " b2 vector dims = (" << N2 << ")" << std::endl;
  std::cout << " b3 vector dims = (" << N3 << ")" << std::endl;

  // Initialize host array and copy it to CUDA device
  random_matrix(matX_h, N0, Ns);
  random_matrix(matY_h, N3, Ns);

  random_matrix(matw1_h, N1, N0);
  random_matrix(matw2_h, N2, N1);
  random_matrix(matw3_h, N3, N2);

  random_vector(vecb1_h, N1);
  random_vector(vecb2_h, N2);
  random_vector(vecb3_h, N3);
  
  // copy data from host to device
  cudaMemcpy(matX_d, matX_h, size_matX, cudaMemcpyHostToDevice);
  cudaMemcpy(matY_d, matY_h, size_matY, cudaMemcpyHostToDevice);

  cudaMemcpy(matw1_d, matw1_h, size_matw1, cudaMemcpyHostToDevice);
  cudaMemcpy(matw2_d, matw2_h, size_matw2, cudaMemcpyHostToDevice);
  cudaMemcpy(matw3_d, matw3_h, size_matw3, cudaMemcpyHostToDevice);

  cudaMemcpy(vecb1_d, vecb1_h, size_vecb1, cudaMemcpyHostToDevice);
  cudaMemcpy(vecb2_d, vecb2_h, size_vecb2, cudaMemcpyHostToDevice);
  cudaMemcpy(vecb3_d, vecb3_h, size_vecb3, cudaMemcpyHostToDevice);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////computation///////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////  

  // forward propagation
  float lr = 0.1;
  compute_nn(lr, matw1_d, matw2_d, matw3_d, vecb1_d, vecb2_d, vecb3_d, matX_d, matY_d, Ns, N0, N1, N2, N3);

  // Retrieve result from device and store it in host array
  cudaMemcpy(matw1_h, matw1_d, size_matw1, cudaMemcpyDeviceToHost);
  cudaMemcpy(matw2_h, matw2_d, size_matw2, cudaMemcpyDeviceToHost);
  cudaMemcpy(matw3_h, matw3_d, size_matw3, cudaMemcpyDeviceToHost);

  cudaMemcpy(vecb1_h, vecb1_d, size_vecb1, cudaMemcpyDeviceToHost);
  cudaMemcpy(vecb2_h, vecb2_d, size_vecb2, cudaMemcpyDeviceToHost);
  cudaMemcpy(vecb3_h, vecb3_d, size_vecb3, cudaMemcpyDeviceToHost);

  // Print results
  print_matrix(matw3_h, N3, N2);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////finalization//////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////    
  
  // Cleanup
  delete [] matX_h;
  delete [] matY_h;
  delete [] matw1_h;
  delete [] matw2_h;
  delete [] matw3_h;
  delete [] vecb1_h;
  delete [] vecb2_h;
  delete [] vecb3_h;

  cudaFree(matX_d);
  cudaFree(matY_d);
  cudaFree(matw1_d);
  cudaFree(matw2_d);
  cudaFree(matw3_d);
  cudaFree(vecb1_d);
  cudaFree(vecb2_d);
  cudaFree(vecb3_d);

  return 0;
}