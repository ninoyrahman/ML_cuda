// Kernel that executes on the CUDA device
__global__ void square_array(float *a, float *b, float *c, int N){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx<N){ 
    c[idx] = a[idx] + b[idx];
  }

}