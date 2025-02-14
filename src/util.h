#include <cstdlib>
#include <ctime>

// random matrix
void random_matrix(float *a, int a_row, int a_col){
  std::srand(std::time({}));
  for (int i=0; i<a_row; i++){ 
    for (int j=0; j<a_col; j++){ 
      a[i * a_row + j] = (float(rand()) / RAND_MAX) - 0.5;
    }
  }  
}

// print matrix
void print_matrix(float *a, int a_row, int a_col){
  for (int i=0; i<a_row; i++){ 
    for (int j=0; j<a_col; j++){ 
      std::cout << "(" << i << "," << j << ") " << a[i * a_row + j] << " ";
    }
    std::cout << std::endl;
  }  
}