/**
 * @file util_nn.h
 * @author ninoy rahman
 * @brief functions for neural network
 * @version 0.1
 * @date 2025-02-16
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "cuda.h"
#include "cublas_v2.h"
#include <iostream>
#include "math.h"
#include <algorithm>
#include <iterator>

/**
 * @brief kernal for ReLU
 * 
 * @param z[in] input array
 * @param a[out] output array
 * @param n[in] size of z and a 
 */
__global__ void ReLU(const float *z, float *a, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = fmaxf(z[idx], 0);
}

/**
 * @brief kernal for derivative of ReLU
 * 
 * @param z[in] input array
 * @param a[out] output array
 * @param n[in] size of z and a 
 */
__global__ void dReLU(const float *z, float *da, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    da[idx] = (float)(z[idx] > 0.0f);
}

/**
 * @brief kernal for clipping
 * 
 * @param clip[in] clip value
 * @param a[in] input array
 * @param a_c[out] output array
 * @param n[in] size of z and a 
 */
__global__ void clipping(float clip, float *a, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = fminf(fmaxf(a[idx], -clip), clip);
}

float get_accuracy(const float *a3_h, const float *Y_h, const int N3, const int Ns){

    int idx_a3;
    int idx_Y;
    float acc = 0.0f;
    // int size = 5;
    
    for (int col = 0; col < Ns; col++){
        idx_a3 = std::distance(a3_h + col * N3, std::max_element(a3_h + col * N3, a3_h + (col + 1) * N3 - 1));
        idx_Y  = std::distance(Y_h  + col * N3, std::max_element(Y_h  + col * N3, Y_h  + (col + 1) * N3 - 1));
        acc += (float)(idx_a3 == idx_Y); // true=1.0, false=0.0
    }
    acc = acc / (float)Ns;

    return acc;
}

/**
 * @brief update weights and biases
 * 
 * @param dw1[in out] gradient of w1
 * @param dw2[in out] gradient of w2 
 * @param dw3[in out] gradient of w3
 * @param db1[in out] gradient of b1
 * @param db2[in out] gradient of b2
 * @param db3[in out] gradient of b3
 * @param Ns[in] sample size
 * @param N0[in] feature size
 * @param N1[in] 1st hidden layer size
 * @param N2[in] 2nd hidden layer size
 * @param N3[in] label size
 */
void gradient_clipping(const float clip, float *dw1, float *dw2, float *dw3, float *db1, float *db2, float *db3,
    const int N0, const int N1, const int N2, const int N3){

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid;

    blocksPerGrid.x = (int)ceil(N1 * N0 / threadsPerBlock.x);
    clipping <<< blocksPerGrid, threadsPerBlock >>> (1.0f, dw1, N1 * N0);

    blocksPerGrid.x = (int)ceil(N2 * N1 / threadsPerBlock.x);
    clipping <<< blocksPerGrid, threadsPerBlock >>> (1.0f, dw2, N2 * N1);

    blocksPerGrid.x = (int)ceil(N3 * N2 / threadsPerBlock.x);
    clipping <<< blocksPerGrid, threadsPerBlock >>> (1.0f, dw3, N3 * N2);

    blocksPerGrid.x = (int)ceil(N1 / threadsPerBlock.x);
    clipping <<< blocksPerGrid, threadsPerBlock >>> (1.0f, db1, N1);

    blocksPerGrid.x = (int)ceil(N2 / threadsPerBlock.x);
    clipping <<< blocksPerGrid, threadsPerBlock >>> (1.0f, db2, N2);

    blocksPerGrid.x = (int)ceil(N3 / threadsPerBlock.x);
    clipping <<< blocksPerGrid, threadsPerBlock >>> (1.0f, db3, N3);

}

/**
 * @brief forward propagation
 * 
 * @param a1[out] activation output at 1st hidden layer
 * @param a2[out] activation output at 2nd hidden layer
 * @param a3[out] activation output at output layer
 * @param z1[out] linear combination at 1st hidden layer
 * @param z2[out] linear combination at 2nd hidden layer
 * @param z3[out] linear combination at output layer
 * @param w1[in] weights at 1st hidden layer
 * @param w2[in] weights at 2nd hidden layer
 * @param w3[in] weights at output layer
 * @param b1[in] biases at 1st hidden layer
 * @param b2[in] biases at 2nd hidden layer 
 * @param b3[in] biases at output layer 
 * @param X[in] input feature
 * @param Vone_d[in] one vector
 * @param a3_h[out] activation output at output layer (host)
 * @param a3_hd[out] activation output at output layer (host, double)
 * @param Ns[in] sample size
 * @param N0[in] feature size
 * @param N1[in] 1st hidden layer size
 * @param N2[in] 2nd hidden layer size
 * @param N3[in] label size
 */
void forward_propagation(float *a1, float *a2, float *a3, float *z1, float *z2, float *z3, 
    const float *w1, const float *w2, const float *w3, const float *b1, const float *b2, const float *b3, 
    const float *X, const float *Vone_d, float *a3_h, double *a3_hd,
    const int Ns, const int N0, const int N1, const int N2, const int N3){

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid;

    cublasStatus_t status;
    cublasHandle_t handle;

    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // printf("forward\n");

    // z1[N1, Ns] = w1[N1, N0] x X[N0, Ns]
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        N1, Ns, N0, &alpha, // m, n, k
        w1, N1, // A(m x k)=w1
        X,  N0, // B(k x n)=X
        &beta, z1, N1); // C(m x n)=z1
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSgemm returned error code %d\n", status);

    // z1[N1, Ns] = b1[N1] * Vone[Ns] + z1[N1, Ns]
    status = cublasSger(handle, 
        N1, Ns, // m, n
        &alpha,
        b1, 1, // x(m)=b1
        Vone_d, 1, // y(n)=Vone
        z1, N1); // A(m x n)=z1
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSger returned error code %d\n", status);
    // printf("w1: "); test_value(w1, N1, N0);
    // printf("X: "); test_value(X, N0, Ns);
    // printf("b1: "); test_value(b1, N1, 1);
    // printf("z1: "); test_value(z1, N1 * Ns);

    // a1[N1, Ns] = ReLU(z1[N1, Ns])
    blocksPerGrid.x = (int)ceil(N1 * Ns / threadsPerBlock.x);
    ReLU <<< blocksPerGrid, threadsPerBlock >>> (z1, a1, N1 * Ns);

    // z2[N2, Ns] = w2[N2, N1] x a1[N1, Ns]
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        N2, Ns, N1, &alpha, // m, n, k
        w2, N2, // A(m x k)=w2 
        a1,  N1, // B(k x n)=a1
        &beta, z2, N2); // C(m x n)=z2
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSger returned error code %d\n", status);

    // z2[N2, Ns] = b2[N2] * Vone[Ns] + z2[N2, Ns]
    status = cublasSger(handle, 
        N2, Ns, // m, n
        &alpha,
        b2, 1, // x(m)=b2
        Vone_d, 1, // y(n)=Vone
        z2, N2); // A(m x n)=z2
    if (status != CUBLAS_STATUS_SUCCESS)
        printf("cublasSger returned error code %d\n", status);
    // printf("w2: "); test_value(w2, N2, N1);
    // printf("a1: "); test_value(a1, N1, Ns);        
    // printf("b2: "); test_value(b2, N2, 1);
    // printf("z2: "); test_value(z2, N2 * Ns);

    // a2[N2, Ns] = ReLU(z2[N2, Ns])
    blocksPerGrid.x = (int)ceil(N2 * Ns / threadsPerBlock.x);
    ReLU <<< blocksPerGrid, threadsPerBlock >>> (z2, a2, N2 * Ns);

    // z3[N3, Ns] = w3[N3, N2] x a2[N2, Ns]
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        N3, Ns, N2, &alpha, // m, n, k
        w3, N3, // A(m x k)=w3 
        a2,  N2, // B(k x n)=a2
        &beta, z3, N3); // C(m x n)=z3
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSger returned error code %d\n", status);

    // z3[N3, Ns] = b3[N3] * Vone[Ns] + z3[N3, Ns]
    status = cublasSger(handle, 
        N3, Ns, // m, n
        &alpha,
        b3, 1, // x(m)=b3
        Vone_d, 1, // y(n)=Vone
        z3, N3); // A(m x n)=z3
    if (status != CUBLAS_STATUS_SUCCESS)
        printf("cublasSger returned error code %d\n", status);
    // printf("w3: "); test_value(w3, N3, N2);
    // printf("a2: "); test_value(a2, N2, Ns);
    // printf("b3: "); test_value(b3, N3, 1);
    // printf("z3: "); test_value(z3, N3 * Ns);

    // a3[N3, Ns] = softmax(z3[N3, Ns])
    cudaMemcpy(a3_h, z3, N3 * Ns * sizeof(float), cudaMemcpyDeviceToHost); // copy from device to host  

    double sum_exp;
    for (int col = 0; col < Ns; col++){
        sum_exp = 0e0;
        for (int row = 0; row < N3; row++){
            a3_hd[col * N3 + row] = exp((double)a3_h[col * N3 + row]);
            sum_exp += a3_hd[col * N3 + row];
            if(a3_h[col * N3 + row] > 709.782f) {
                // printf("z3: "); test_value(z3, N3 * Ns);
                perror("overflow while exp compute");
                exit(1);
            }
        }
        sum_exp = max(sum_exp, 1e-6);
        for (int row = 0; row < N3; row++){
            a3_h[col * N3 + row] = (float)(a3_hd[col * N3 + row] / sum_exp);
            if(a3_h[col * N3 + row] > 1.0f) {
                perror("softmax probability is greater than one");
                exit(1);
            }            
        }
    }

    cudaMemcpy(a3, a3_h, N3 * Ns * sizeof(float), cudaMemcpyHostToDevice);
    // printf("a3: "); test_value(a3, N3, Ns);

    cublasDestroy(handle);

}

/**
 * @brief 
 * 
 * @param dw1[out] gradient of w1
 * @param dw2[out] gradient of w2 
 * @param dw3[out] gradient of w3
 * @param db1[out] gradient of b1
 * @param db2[out]  gradient of b2
 * @param db3[out]  gradient of b3
 * @param w1[in] weights at 1st hidden layer
 * @param w2[in] weights at 2nd hidden layer
 * @param w3[in] weights at output layer
 * @param a1[in] activation output at 1st hidden layer
 * @param a2[in] activation output at 2nd hidden layer
 * @param a3[in] activation output at output layer
 * @param z1[in] linear combination at 1st hidden layer
 * @param z2[in] linear combination at 2nd hidden layer
 * @param z3[in] linear combination at output layer
 * @param X[in] features 
 * @param Y[in] labels
 * @param Vone_d[in] one Vector
 * @param delta[out] error at output layer
 * @param delta1[out] error at 2nd hidden layer
 * @param delta2[out] error at 1st hidden layer
 * @param tmp1[out] temporary variable
 * @param tmp2[out] temporary variable
 * @param dz1[out] derivative of ReLU at 2nd hidden layer
 * @param dz2[out] derivative of ReLU at 1st hidden layer
 * @param Ns[in] sample size
 * @param N0[in] feature size
 * @param N1[in] 1st hidden layer size
 * @param N2[in] 2nd hidden layer size
 * @param N3[in] label size
 */
void backward_propagation(float *dw1, float *dw2, float *dw3, float *db1, float *db2, float *db3,
    const float *w1, const float *w2, const float *w3,
    const float *a1, const float *a2, const float *a3, const float *z1, const float *z2, const float *z3,
    const float *X, const float *Y, const float *Vone_d, 
    float *delta, float *delta1, float *delta2,
    float *tmp1, float *tmp2, float *dz1, float *dz2,
    const int Ns, const int N0, const int N1, const int N2, const int N3){

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid;
    
    cublasStatus_t status;
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha;
    float beta;

    // printf("backward\n");

    // delta[N3, Ns] = a3[N3, Ns]
    status = cublasScopy(handle, N3 * Ns, a3, 1, delta, 1);
    if (status != CUBLAS_STATUS_SUCCESS)
        printf("cublasScopy delta[N3, Ns] = a3[N3, Ns] returned error code %d\n", status);      

    // delta[N3, Ns] = (1.0 / Ns) * (delta=a3[N3, Ns] - Y[N3, Ns])
    alpha = -1.0f;
    status = cublasSaxpy(handle, N3 * Ns, &alpha, Y, 1, delta, 1);
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSaxpy delta[N3, Ns] = delta=a3[N3, Ns] - Y[N3, Ns] returned error code %d\n", status);
    
    alpha = 1.0f / float(Ns);
    status = cublasSscal(handle, N3 * Ns, &alpha, delta, 1);
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSscal delta[N3, Ns] = (1.0 / Ns) * delta[N3, Ns] returned error code %d\n", status);  

    // dw3[N3, N2] = delta[N3, Ns] * a2[N2, Ns].T
    alpha = 1.0f;
    beta = 0.0f;
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        N3, N2, Ns, // m, n, k
        &alpha,
        delta, N3, // A(m x k)=delta
        a2, N2, // B(n x k)=a2
        &beta,
        dw3, N3); // C(m x n)=dw3
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSgemm dw3[N3, N2] = delta[N3, Ns] * a2[N2, Ns].T returned error code %d\n", status);
    // printf("delta: "); test_value(delta, N3 * Ns);
    // printf("a2: "); test_value(a2, N2 * Ns);
    // printf("dw3: "); test_value(dw3, N3 * N2);
    
    // db3[N3] = delta[N3, Ns].sum(column) = delta[N3, Ns] * Vone[Ns]
    status = cublasSgemv(handle, CUBLAS_OP_N,
        N3, Ns, // m, n
        &alpha,
        delta, N3, // A(m x n)=delta
        Vone_d, 1, // x(n)=one
        &beta,
        db3, 1); // y(m)=db3
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSgemv db3[N3] = delta[N3, Ns].sum(column) = delta[N3, Ns] * Vone[Ns] returned error code %d\n", status);
    // printf("db3: "); test_value(db3, N3);
    
    // delta1 = w3.T.dot(delta) * self.dReLU(z2)
    // tmp1[N2, Ns] = w3[N3, N2].T * delta[N3, Ns] 
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        N2, Ns, N3, // m, n, k
        &alpha,
        w3, N3, // A(k x m)=w3
        delta, N3, // B(k x n)=delta
        &beta,
        tmp1, N2); // C(m x n)=tmp1 
    if (status != CUBLAS_STATUS_SUCCESS)
        printf("cublasSgemm tmp1[N2, Ns] = w3[N3, N2].T * delta[N3, Ns] returned error code %d\n", status);
    
    // delta1[N2, Ns] = tmp1[N2, Ns] * self.dReLU(z2)[N2, Ns]
    blocksPerGrid.x = (int)ceil(N2 * Ns / threadsPerBlock.x);
    dReLU <<< blocksPerGrid, threadsPerBlock >>> (z2, dz1, N2 * Ns);

    status = cublasSdgmm(handle, CUBLAS_SIDE_LEFT, 
        N2 * Ns, 1, // m, n
        tmp1, N2 * Ns, // A(m x n)=tmp1
        dz1, 1, // X(1 x m)=z2
        delta1, N2 * Ns); // C(m x n)=delta1
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSdgmm delta1[N2, Ns] = delta1[N2, Ns] * self.dfact(z2)[N2, Ns] returned error code %d\n", status);

    // dw2[N2, N1] = delta1[N2, Ns] * a1[N1, Ns].T
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        N2, N1, Ns, // m, n, k
        &alpha,
        delta1, N2, // A(m x k)=delta1
        a1, N1, // B(n x k)=a1
        &beta,
        dw2, N2); // C(m x n)=dw2
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSgemm dw2[N2, N1] = delta1[N2, Ns] * a1[N1, Ns].T returned error code %d\n", status);
    // printf("delta1: "); test_value(delta1, N2 * Ns);
    // printf("a1: "); test_value(a1, N1 * Ns);
    // printf("dw2: "); test_value(dw2, N2 * N1);

    // db2[N2] = delta1[N2, Ns].sum(column) = delta1[N2, Ns] * Vone[Ns]
    status = cublasSgemv(handle, CUBLAS_OP_N,
        N2, Ns, // m, n
        &alpha,
        delta1, N2, // A(m x n)=delta1
        Vone_d, 1, // x(n)=one
        &beta,
        db2, 1); // y(m)=db2
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSgemv db2[N2] = delta1[N2, Ns] * Vone[Ns] returned error code %d\n", status);
    // printf("db2: "); test_value(db2, N2);

    // delta2 = w2.T.dot(delta1) * self.dReLU(z1)
    // tmp2[N1, Ns] = w2[N2, N1].T * delta1[N2, Ns]
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        N1, Ns, N2, // m, n, k
        &alpha,
        w2, N2, // A(k x m)=w2
        delta1, N2, // B(k x n)=delta1
        &beta,
        tmp2, N1); // C(m x n)=tmp2
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSgemm tmp1[N2, Ns] = w3[N3, N2].T * delta[N3, Ns] returned error code %d\n", status);

    // delta2[N1, Ns] = tmp2[N1, Ns] * self.dReLU(z1)[N1, Ns]
    blocksPerGrid.x = (int)ceil(N1 * Ns / threadsPerBlock.x);
    dReLU <<< blocksPerGrid, threadsPerBlock >>> (z1, dz2, N1 * Ns);

    status = cublasSdgmm(handle, CUBLAS_SIDE_LEFT, 
        N1 * Ns, 1, // m, n
        tmp2, N1 * Ns, // A(m x n)=tmp2
        dz2, 1, // X(1 x m)=z1
        delta2, N1 * Ns); // C(m x n)=delta2
    if (status != CUBLAS_STATUS_SUCCESS)
        printf("cublasSdgmm delta2[N1, Ns] = tmp2[N1, Ns] * self.dfactivation(z1)[N1, Ns] returned error code %d\n", status);

    // dw1[N1, N0] = delta2[N1, Ns] * X[N0, Ns].T
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        N1, N0, Ns, // m, n, k
        &alpha,
        delta2, N1, // A(m x k)=delta2
        X, N0, // B(n x k)=X
        &beta,
        dw1, N1); // C(m x n)=dw1
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSgemm dw1[N1, N0] = delta2[N1, Ns] * X[N0, Ns].T returned error code %d\n", status);
    // printf("delta2: "); test_value(delta2, N1 * Ns);
    // printf("X: "); test_value(X, N0 * Ns);
    // printf("dw1: "); test_value(dw1, N1 * N0);

    // db1[N1] = delta2[N1, Ns].sum(column) = delta2[N1, Ns] * Vone[Ns]
    status = cublasSgemv(handle, CUBLAS_OP_N,
        N1, Ns, // m, n
        &alpha,
        delta2, N1, // A(m x n)=delta1
        Vone_d, 1, // x(n)=one
        &beta,
        db1, 1); // y(m)=db2
    if (status != CUBLAS_STATUS_SUCCESS)
        printf("cublasSgemv db1[N1] = delta2[N1, Ns] * Vone[Ns] returned error code %d\n", status);
    // printf("db1: "); test_value(db1, N1);

    cublasDestroy(handle);

}

/**
 * @brief update weights and biases
 * 
 * @param w1[out] weights at 1st hidden layer
 * @param w2[out] weights at 2nd hidden layer
 * @param w3[out] weights at output layer
 * @param b1[out] biases at 1st hidden layer
 * @param b2[out] biases at 2nd hidden layer 
 * @param b3[out] biases at output layer
 * @param lr learning rate
 * @param dw1[in] gradient of w1
 * @param dw2[in] gradient of w2 
 * @param dw3[in] gradient of w3
 * @param db1[in] gradient of b1
 * @param db2[in]  gradient of b2
 * @param db3[in]  gradient of b3
 * @param Ns[in] sample size
 * @param N0[in] feature size
 * @param N1[in] 1st hidden layer size
 * @param N2[in] 2nd hidden layer size
 * @param N3[in] label size
 */
void update_parameter(float *w1, float *w2, float *w3, float *b1, float *b2, float *b3,
    const float lr, float *dw1, float *dw2, float *dw3, float *db1, float *db2, float *db3, 
    const int Ns, const int N0, const int N1, const int N2, const int N3){

    cublasStatus_t status;
    cublasHandle_t handle;

    cublasCreate(&handle);

    float alpha = -lr;

    // printf("update\n");

    // gradient_clipping(1.0f, dw1, dw2, dw3, db1, db2, db3, N0, N1, N2, N3);
    // printf("w1: "); test_value(w1, N1 * N0);

    // w1[N1, N0] = w1[N1, N0] - lr * dw1[N1, N0]
    status = cublasSaxpy(handle, N1 * N0, &alpha, dw1, 1, w1, 1);
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSaxpy w1[N1, N0] = w1[N1, N0] - lr * dw1[N1, N0] returned error code %d\n", status);
    // printf("w1: "); test_value(w1, N1 * N0);

    // b1[N1] = b1[N1] - lr * db1[N1]
    status = cublasSaxpy(handle, N1, &alpha, db1, 1, b1, 1);
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSaxpy b1[N1] = b1[N1] - lr * db1[N1] returned error code %d\n", status);

    // w2[N2, N1] = w2[N2, N1] - lr * dw2[N2, N1]
    status = cublasSaxpy(handle, N2 * N1, &alpha, dw2, 1, w2, 1);
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSaxpy w2[N2, N1] = w2[N2, N1] - lr * dw2[N2, N1] returned error code %d\n", status);

    // b2[N2] = b2[N2] - lr * db2[N2]
    status = cublasSaxpy(handle, N2, &alpha, db2, 1, b2, 1);
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSaxpy b2[N2] = b2[N2] - lr * db2[N2] returned error code %d\n", status);

    // w3[N3, N2] = w3[N3, N2] - alpha * dw3[N3, N2]
    status = cublasSaxpy(handle, N3 * N2, &alpha, dw3, 1, w3, 1);
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSaxpy w3[N3, N2] = w3[N3, N2] - alpha * dw3[N3, N2] returned error code %d\n", status);

    // b3[N3] = b3[N3] - alpha * db3[N3]
    status = cublasSaxpy(handle, N3, &alpha, db3, 1, b3, 1);
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSaxpy b3[N3] = b3[N3] - alpha * db3[N3] returned error code %d\n", status);

    cublasDestroy(handle);

}