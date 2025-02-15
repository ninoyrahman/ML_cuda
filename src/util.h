#include <cstdlib>
#include <ctime>
#include <cassert>

// random matrix
void random_matrix(float *a, int a_row, int a_col){
  std::srand(std::time({}));
  for (int i=0; i<a_row; i++){ 
    for (int j=0; j<a_col; j++){ 
      a[i * a_col + j] = (float(rand()) / RAND_MAX) - 0.5;
    }
  }  
}

// transpose matrix
void transpose_matrix(const float *a, float *a_t, int a_row, int a_col){
  for (int row=0; row<a_row; row++){ 
    for (int col=0; col<a_col; col++){ 
      a_t[col * a_row + row] = a[row * a_col + col];
    }
  }  
}

// print matrix
void print_matrix(const float *a, int a_row, int a_col){
  for (int i=0; i<a_row; i++){ 
    for (int j=0; j<a_col; j++){ 
      std::cout << "(" << i << "," << j << ") " << a[i * a_col + j] << " ";
    }
    std::cout << std::endl;
  }  
}

// Kernel for matrix multiplication C = A * B, b_row = a_col, b_col = c_col
void mat_mul_varify(const float *a, const float *b, const float *c, int c_row, int c_col, int b_row){
    
  float tmp;
  float diff = 0.0;

  for (int row = 0; row < c_row; row++){
    for (int col = 0; col < c_col; col++){
      tmp = 0.0;
      for (int i = 0; i < b_row; i++){
        tmp += a[row * b_row + i] * b[i * c_col + col];
      }
      // printf("%d %d %f %f", row, col, tmp, c[row * c_col + col]);
      diff = max(max(std::fabs(tmp - c[row * c_col + col]), 1e-6), diff);
    }
  }
  printf("max diff = %f\n", diff);

}
 
// rearrange output matrix of cublasSgemm
void rearrange_matrix(const float *c, float *c_r, int c_row, int c_col){
  for (int row = 0; row < c_row; row++){
    for (int col = 0; col < c_col; col++){
      c_r[row * c_col + col] = c[col * c_row + row];
    }
  }  
}