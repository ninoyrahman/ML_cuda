/**
 * @file nn.h
 * @author ninoy rahman
 * @brief neural network routines
 * @version 0.1
 * @date 2025-02-15
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <iostream>
#include "cuda.h"
#include "util_nn.h"

/**
 * @brief perform neural network
 * 
 * @param lr learning rate
 * @param epoch max number of epoch
 * @param w1[in out] weights from input layer to first hidden layer
 * @param w2[in out] weights from first hidden layer to second hidden layer
 * @param w3[in out] weights from second hidden layer to output layer
 * @param b1[in out] biases from input layer to first hidden layer
 * @param b2[in out] biases from first hidden layer to second hidden layer
 * @param b3[in out] biases from second hidden layer to output layer
 * @param X[in] input training data
 * @param Y[in] output training data
 * @param Y_h[in] output training data (host)
 * @param Ns[in] sample size
 * @param N0[in] input feature size
 * @param N1[in] first hidden layer size
 * @param N2[in] second hidden layer size
 * @param N3[in] output label size
 */
void compute_nn(float lr, int epoch, float *w1, float *w2, float *w3, float *b1, float *b2, float *b3, float *X, float *Y, float *Y_h, int Ns, int N0, int N1, int N2, int N3){

    // Pointer to device arrays
    float *a1, *a2, *a3;
    float *z1, *z2, *z3;
    float *dw1, *dw2, *dw3;
    float *db1, *db2, *db3;

    float acc;

    // size of vector
    size_t size_a1 = N1 * Ns * sizeof(float);
    size_t size_a2 = N2 * Ns * sizeof(float);
    size_t size_a3 = N3 * Ns * sizeof(float);

    // allocate on device
    cudaMalloc((void **) &a1, size_a1);
    cudaMalloc((void **) &a2, size_a2);
    cudaMalloc((void **) &a3, size_a3);
    cudaMalloc((void **) &z1, size_a1);
    cudaMalloc((void **) &z2, size_a2);
    cudaMalloc((void **) &z3, size_a3);

    cudaMalloc((void **) &dw1, N1 * N0 * sizeof(float));
    cudaMalloc((void **) &dw2, N2 * N1 * sizeof(float));
    cudaMalloc((void **) &dw3, N3 * N2 * sizeof(float));

    cudaMalloc((void **) &db1, N1 * sizeof(float));
    cudaMalloc((void **) &db2, N2 * sizeof(float));
    cudaMalloc((void **) &db3, N3 * sizeof(float));

    for (int i = 0; i < epoch; i++){

        // if(i*10%epoch == 0) printf("===");
        
        // forward propagationx
        forward_propagation(a1, a2, a3, z1, z2, z3, 
            w1, w2, w3, b1, b2, b3, X, 
            Ns, N0, N1, N2, N3);

        // backward propagation
        backward_propagation(dw1, dw2, dw3, db1, db2, db3,
            w1, w2, w3,
            a1, a2, a3, z1, z2, z3,
            X, Y,
            Ns, N0, N1, N2, N3);
            
        // update weights and biases
        update_parameter(w1, w2, w3, b1, b2, b3,
            lr, dw1, dw2, dw3, db1, db2, db3, 
            Ns, N0, N1, N2, N3);

        if (i % 100 == 0){
            acc = get_accuracy(a3, Y_h, N3, Ns);
            printf("i=%d, accuracy=%.3f\n", i, acc);
        }
    }

    forward_propagation(a1, a2, a3, z1, z2, z3, 
        w1, w2, w3, b1, b2, b3, X, 
        Ns, N0, N1, N2, N3);
    printf("accuracy=%.3f\n", get_accuracy(a3, Y_h, N3, Ns));

    // free memory on device
    cudaFree(a1); cudaFree(a2); cudaFree(a3);
    cudaFree(z1); cudaFree(z2); cudaFree(z3);
    cudaFree(dw1); cudaFree(dw2); cudaFree(dw3);
    cudaFree(db1); cudaFree(db2); cudaFree(db3);

}