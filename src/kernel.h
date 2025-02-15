#include "cublas_v2.h"
#include <iostream>

// Kernel that print vector
__global__ void print_vector_device(const float *a, int n){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n){
    printf("(%d) %f ", idx, a[idx]);
  }
  
}

// Kernel that executes on the CUDA device
__global__ void sum_array(float *a, float *b, float *c, int N){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
  if (idx<N){ 
    c[idx] = a[idx] + b[idx];
  }
  
}
  
// Kernel for matrix multiplication C = A * B, b_row = a_col, b_col = c_col 
__global__ void mat_mul(const float *a, const float *b, float *c, int c_row, int c_col, int b_row){
  
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  
  float tmp = 0;
  for (int i = 0; i < b_row; i++){
    tmp += a[row * b_row + i] * b[i * c_col + col];
  }
  c[row * c_col + col] = tmp;

}

// Kernel for matrix multiplication C = AT * B
__global__ void mat_mul_transpose(const float *a, const float *b, float *c, int c_row, int c_col, int b_row){
  
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  
  float tmp = 0;
  for (int i = 0; i < b_row; i++){
    tmp += a[i * c_row + row] * b[i * c_col + col];
  }
  c[row * c_col + col] = tmp;

}

void compute(const float *matA, const float *matB, float *matC, float *vecB, float *vecO, int N1, int N2, int N3){

  cublasStatus_t status;
  cublasHandle_t handle;
    
  float alpha = 1.0f;
  float beta = 0.0f;

  cublasCreate(&handle);

  status = cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N1, N2, N3,
    &alpha,
    matA, N1,
    matB, N3,
    &beta,
    matC, N1);

  if (status != CUBLAS_STATUS_SUCCESS) 
    printf("cublasSgemm returned error code %d\n", status);

  // gpu matrix vector addition
  alpha = 1.0f;
  status = cublasSger(handle, 
    N1, N2,
    &alpha,
    vecB, 1,
    vecO, 1,
    matC, N1);

  if (status != CUBLAS_STATUS_SUCCESS) 
    printf("cublasSger returned error code %d\n", status);

}