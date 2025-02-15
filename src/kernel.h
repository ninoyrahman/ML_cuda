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