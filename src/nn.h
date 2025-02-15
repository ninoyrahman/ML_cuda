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


#include "cublas_v2.h"
#include <iostream>
/**
 * @brief perform neural network
 * 
 * @param w1[in out] weights from input layer to first hidden layer
 * @param w2[in out] weights from first hidden layer to second hidden layer
 * @param w3[in out] weights from second hidden layer to output layer
 * @param b1[in out] biases from input layer to first hidden layer
 * @param b2[in out] biases from first hidden layer to second hidden layer
 * @param b3[in out] biases from second hidden layer to output layer
 * @param X[in] input training data
 * @param Y[in] output training data
 * @param Ns[in] sample size
 * @param N0[in] input feature size
 * @param N1[in] first hidden layer size
 * @param N2[in] second hidden layer size
 * @param N3[in] output label size
 */
void compute_nn(float *w1, float *w2, float *w3, float *b1, float *b2, float *b3, float *X, float *Y, int Ns, int N0, int N1, int N2, int N3){

    cublasStatus_t status;
    cublasHandle_t handle;
    
    float alpha;
    float beta;

    // Pointer to device arrays
    float *a1, *a2, *a3;
    float *z1, *z2, *z3;

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

    // Vone on host
    float *Vone_h  = new float[Ns];
    for (int i = 0; i < Ns; i++)
        Vone_h[i] = 1.0f;

    // Vone on device
    float *Vone_d;
    size_t size_Vone = Ns * sizeof(float);
    cudaMalloc((void **) &Vone_d, size_Vone);
    cudaMemcpy(Vone_d, Vone_h, size_Vone, cudaMemcpyHostToDevice);

    cublasCreate(&handle);

    // forward propagation

    // a1[N1, Ns] = w1[N1, N0] x X[N0, Ns]
    alpha = 1.0f;
    beta = 0.0f;
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        N1, Ns, N0, &alpha,
        w1, N1, // A=w1
        X,  N0, // B=X
        &beta, a1, N1); // C=a1
    
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSgemm returned error code %d\n", status);

    // a1[N1, Ns] = b1[N1] * Vone[Ns] + a1[N1, Ns]
    status = cublasSger(handle, 
        N1, Ns,
        &alpha,
        b1, 1,
        Vone_d, 1,
        a1, N1);

    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSger returned error code %d\n", status);


    // a2[N2, Ns] = w2[N2, N1] x a1[N1, Ns]
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        N2, Ns, N1, &alpha,
        w2, N2, // A=w2
        a1,  N1, // B=a1
        &beta, a2, N2); // C=a2
    
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSger returned error code %d\n", status);        

    // a2[N2, Ns] = b2[N2] * Vone[Ns] + a2[N2, Ns]
    status = cublasSger(handle, 
        N2, Ns,
        &alpha,
        b2, 1,
        Vone_d, 1,
        a2, N2);
    
    if (status != CUBLAS_STATUS_SUCCESS)
        printf("cublasSger returned error code %d\n", status);

    // a3[N3, Ns] = w3[N3, N2] x a2[N2, Ns]
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        N3, Ns, N2, &alpha,
        w3, N3, // A=w3
        a2,  N2, // B=a2
        &beta, a3, N3); // C=a3
    
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSger returned error code %d\n", status);
    
    // a3[N3, Ns] = b3[N3] * Vone[Ns] + a3[N3, Ns]
    status = cublasSger(handle, 
        N3, Ns,
        &alpha,
        b3, 1,
        Vone_d, 1,
        a3, N3);
    
    if (status != CUBLAS_STATUS_SUCCESS)
        printf("cublasSger returned error code %d\n", status);

    // backward propagation

    cudaFree(a1); cudaFree(a2); cudaFree(a3);
    cudaFree(z1); cudaFree(z2); cudaFree(z3);
    delete [] Vone_h; cudaFree(Vone_d);

}