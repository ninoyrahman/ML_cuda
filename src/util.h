#include <cstdlib>
#include <ctime>
#include <cassert>

// random matrix (column first)
void random_matrix(float *a, const int a_row, const int a_col){
  std::srand(std::time({}));
  for (int j=0; j<a_col; j++){
    for (int i=0; i<a_row; i++){ 
      // a[i * a_col + j] = (float(rand()) / RAND_MAX) - 0.5;
      a[j * a_row + i] = (float(rand()) / RAND_MAX) - 0.5;
    }
  }
}

// random vector
void random_vector(float *a, const int n){
  std::srand(std::time({}));
  for (int i=0; i<n; i++){    
    a[i] = (float(rand()) / RAND_MAX) - 0.5;
  }
}

// vector with single value for all element
void setvalue_vector(float *a, const int n, const float value){
  std::srand(std::time({}));
  for (int i=0; i<n; i++){    
    a[i] = value;
  }
}

// print vector
void print_vector(float *a, const int n){
  std::srand(std::time({}));
  for (int i=0; i<n; i++){    
    printf("(%d) %f ", i, a[i]);
  }
  printf("\n");
}

// transpose matrix
void transpose_matrix(const float *a, float *a_t, int a_row, int a_col){
  for (int row=0; row<a_row; row++){ 
    for (int col=0; col<a_col; col++){ 
      a_t[col * a_row + row] = a[row * a_col + col];
    }
  }  
}

// print matrix (host)
void print_matrix(const float *a, int a_row, int a_col){
  for (int i=0; i<a_row; i++){
    for (int j=0; j<a_col; j++){
      std::cout << "(" << i << "," << j << ") " << a[j * a_row + i] << " ";
    }
    std::cout << std::endl;
  }  
}

// Kernel for matrix multiplication C = A * B, b_row = a_col, b_col = c_col
void mat_mul_varify(const float *a, const float *b, const float *c, int c_row, int c_col, int b_row){
    
  float tmp;
  float diff = 0.0;

  for (int col = 0; col < c_col; col++){
    for (int row = 0; row < c_row; row++){
      tmp = 0.0;
      for (int i = 0; i < b_row; i++){
        // tmp += a[row * b_row + i] * b[i * c_col + col];
        tmp += a[i * c_row + row] * b[col * b_row + i];
      }
      // printf("%d %d %f %f\n", row, col, tmp, c[row * c_col + col]);
      // diff = max(max(std::fabs(tmp - c[row * c_col + col]), 1e-6), diff);
      // printf("%d %d %f %f\n", row, col, tmp, c[col * c_row + row]);
      diff = max(max(std::fabs(tmp - c[col * c_row + row]), 1e-6), diff);
    }
  }
  printf("max diff = %f\n", diff);
  
}

// Kernel for matrix multiplication C = A * B + b, b_row = a_col, b_col = c_col
void mat_mul_vec_sum_varify(const float *a, const float *b, const float *c, const float *vecb, int c_row, int c_col, int b_row){
    
  float tmp;
  float diff = 0.0;

  for (int col = 0; col < c_col; col++){
    for (int row = 0; row < c_row; row++){
      tmp = vecb[row];
      for (int i = 0; i < b_row; i++){
        tmp += a[i * c_row + row] * b[col * b_row + i];
      }
      // printf("%d %d %f %f\n", row, col, tmp + vecb[row], c[col * c_row + row]);
      diff = max(max(std::fabs(tmp - c[col * c_row + row]), 1e-6), diff);
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