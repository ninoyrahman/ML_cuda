// Kernel that executes on the CUDA device
__global__ void sum_array(float *a, float *b, float *c, int N){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx<N){ 
      c[idx] = a[idx] + b[idx];
    }
  
  }
  
  // Kernel for matrix multiplication
  __global__ void mat_mul(float *a, float *b, float *c, int c_row, int c_col, int b_row){
  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (row < c_row && col < c_col){
      c[row * c_row + col] = 0.0;
      for (int i = 0; i < b_row; i++){
        c[row * c_row + col] += a[row * c_row + i] * b[i * b_row + col];
      }
    }
  }