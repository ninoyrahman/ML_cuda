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

/**
 * @brief kernal for ReLU
 * 
 * @param z[in] input array
 * @param a[out] output array
 * @param n[in] size of z and a 
 */
__global__ void ReLU(const float *z, float *a, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
        a[idx] = fmaxf(z[idx], 0);
}

/**
 * @brief kernal for softmax
 * 
 * @param z[in] input array
 * @param a[out] output array
 * @param n[in] size of z and a 
 */
__global__ void softmax(const float *z, float *a, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
        a[idx] = exp(z[idx]);
}

/**
 * @brief kernal for divide by sum(exp(z))
 * 
 * @param a[in out] array with size (m, n)=(N3, Ns)
 * @param sumexp[in] input array with size n
 * @param m[in] size 
 * @param n[in] size 
 */
__global__ void softmax2(const float *tmp, const float *sumexp, float *a, int m, int n){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (tmp[col * m + row] > fmaxf(sumexp[col], 1e-6))
        printf("(%d, %d) %f %f", row, col, tmp[col * m + row], fmaxf(sumexp[col], 1e-6));
    a[col * m + row] = tmp[col * m + row] / fmaxf(sumexp[col], 1e-6);
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
 * @param Ns[in] sample size
 * @param N0[in] feature size
 * @param N1[in] 1st hidden layer size
 * @param N2[in] 2nd hidden layer size
 * @param N3[in] label size
 */
void forward_propagation(float *a1, float *a2, float *a3, float *z1, float *z2, float *z3, 
    const float *w1, const float *w2, const float *w3, const float *b1, const float *b2, const float *b3, const float *X, 
    const int Ns, const int N0, const int N1, const int N2, const int N3){

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid;

    dim3 threadsPerBlock2(8, 16);
    dim3 blocksPerGrid2;

    cublasStatus_t status;
    cublasHandle_t handle;

    // Vone on host
    float *Vone_h  = new float[Ns];
    for (int i = 0; i < Ns; i++)
        Vone_h[i] = 1.0f;
    
    // Vone on device
    float *Vone_d;
    size_t size_Vone = Ns * sizeof(float);
    cudaMalloc((void **) &Vone_d, size_Vone);
    cudaMemcpy(Vone_d, Vone_h, size_Vone, cudaMemcpyHostToDevice);

    float *Vone1_h  = new float[N3];
    for (int i = 0; i < N3; i++)
        Vone1_h[i] = 1.0f;

    float *Vone1_d;
    size_t size_Vone1 = N3 * sizeof(float);
    cudaMalloc((void **) &Vone1_d, size_Vone1);
    cudaMemcpy(Vone1_d, Vone1_h, size_Vone1, cudaMemcpyHostToDevice);

    float *sumexp;
    cudaMalloc((void **) &sumexp, Ns * sizeof(float));

    float *tmp3;
    cudaMalloc((void **) &tmp3, N3 * Ns * sizeof(float));

    cublasCreate(&handle);    

    float alpha = 1.0f;
    float beta = 0.0f;

    // z1[N1, Ns] = w1[N1, N0] x X[N0, Ns]
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        N1, Ns, N0, &alpha,
        w1, N1, // A=w1
        X,  N0, // B=X
        &beta, z1, N1); // C=z1
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSgemm returned error code %d\n", status);

    // z1[N1, Ns] = b1[N1] * Vone[Ns] + z1[N1, Ns]
    status = cublasSger(handle, 
        N1, Ns,
        &alpha,
        b1, 1,
        Vone_d, 1,
        z1, N1);
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSger returned error code %d\n", status);

    // a1[N1, Ns] = ReLU(z1[N1, Ns])
    blocksPerGrid.x = (int)ceil(N1 * Ns / threadsPerBlock.x);
    ReLU <<< blocksPerGrid, threadsPerBlock >>> (z1, a1, N1 * Ns);

    // z2[N2, Ns] = w2[N2, N1] x a1[N1, Ns]
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        N2, Ns, N1, &alpha,
        w2, N2, // A=w2
        a1,  N1, // B=a1
        &beta, z2, N2); // C=z2
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSger returned error code %d\n", status);        

    // z2[N2, Ns] = b2[N2] * Vone[Ns] + z2[N2, Ns]
    status = cublasSger(handle, 
        N2, Ns,
        &alpha,
        b2, 1,
        Vone_d, 1,
        z2, N2);
    if (status != CUBLAS_STATUS_SUCCESS)
        printf("cublasSger returned error code %d\n", status);

    // a2[N2, Ns] = ReLU(z2[N2, Ns])
    blocksPerGrid.x = (int)ceil(N2 * Ns / threadsPerBlock.x);
    ReLU <<< blocksPerGrid, threadsPerBlock >>> (z2, a2, N2 * Ns);

    // z3[N3, Ns] = w3[N3, N2] x a2[N2, Ns]
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        N3, Ns, N2, &alpha,
        w3, N3, // A=w3
        a2,  N2, // B=a2
        &beta, z3, N3); // C=z3
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSger returned error code %d\n", status);

    // z3[N3, Ns] = b3[N3] * Vone[Ns] + z3[N3, Ns]
    status = cublasSger(handle, 
        N3, Ns,
        &alpha,
        b3, 1,
        Vone_d, 1,
        z3, N3);
    if (status != CUBLAS_STATUS_SUCCESS)
        printf("cublasSger returned error code %d\n", status);

    // tmp3[N3, Ns] = softmax(z3[N3, Ns])
    blocksPerGrid.x = (int)ceil(N3 * Ns / threadsPerBlock.x);
    softmax <<< blocksPerGrid, threadsPerBlock >>> (z3, tmp3, N3 * Ns);

    // sumexp[Ns] = tmp3[N3, Ns].T * Vone1[N3]
    status = cublasSgemv(handle, CUBLAS_OP_T,
        N3, Ns, // m, n
        &alpha,
        tmp3, N3, // A(m, n)=tmp3
        Vone1_d, 1, // x(m)=Vone1
        &beta,
        sumexp, 1); // y(n)=sumexp
    if (status != CUBLAS_STATUS_SUCCESS)
        printf("cublasSger returned error code %d\n", status);        

    // a3[N3, Ns] = softmax2(tmp3[N3, Ns])
    blocksPerGrid2.x = (int)ceil(N3 / threadsPerBlock2.x);
    blocksPerGrid2.y = (int)ceil(Ns / threadsPerBlock2.y);
    softmax2 <<< blocksPerGrid2, threadsPerBlock2 >>> (tmp3, sumexp, a3, N3, Ns);

    // free memory
    delete [] Vone_h; cudaFree(Vone_d);
    delete [] Vone1_h; cudaFree(Vone1_d);
    cudaFree(tmp3); cudaFree(sumexp);
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
 * @param Ns[in] sample size
 * @param N0[in] feature size
 * @param N1[in] 1st hidden layer size
 * @param N2[in] 2nd hidden layer size
 * @param N3[in] label size
 */
void backward_propagation(float *dw1, float *dw2, float *dw3, float *db1, float *db2, float *db3,
    const float *w1, const float *w2, const float *w3,
    const float *a1, const float *a2, const float *a3, const float *z1, const float *z2, const float *z3,
    const float *X, const float *Y,
    const int Ns, const int N0, const int N1, const int N2, const int N3){
    
    cublasStatus_t status;
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *delta, *delta1, *delta2;
    float *tmp1, *tmp2;    

    size_t size_a1 = N1 * Ns * sizeof(float);
    size_t size_a2 = N2 * Ns * sizeof(float);
    size_t size_a3 = N3 * Ns * sizeof(float);

    cudaMalloc((void **) &delta,  size_a3);
    cudaMalloc((void **) &delta1, size_a2); cudaMalloc((void **) &tmp1, size_a2);
    cudaMalloc((void **) &delta2, size_a1); cudaMalloc((void **) &tmp2, size_a1);

    // Vone on host
    float *Vone_h  = new float[Ns];
    for (int i = 0; i < Ns; i++)
        Vone_h[i] = 1.0f;
    
    // Vone on device
    float *Vone_d;
    size_t size_Vone = Ns * sizeof(float);
    cudaMalloc((void **) &Vone_d, size_Vone);
    cudaMemcpy(Vone_d, Vone_h, size_Vone, cudaMemcpyHostToDevice);    

    float alpha;
    float beta;

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
    
    // delta1 = w3.T.dot(delta) * self.dfactivation(z2)
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
    
    // delta1[N2, Ns] = tmp1[N2, Ns] * self.dfactivation(z2)[N2, Ns]
    status = cublasSdgmm(handle, CUBLAS_SIDE_LEFT, 
        N2 * Ns, 1, // m, n
        tmp1, N2 * Ns, // A(m x n)=tmp1
        z2, 1, // X(1 x m)=z2
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

    // delta2 = w2.T.dot(delta1) * self.dfactivation(z1)
    // tmp2[N1, Ns] = w2[N2, N1].T * delta1[N2, Ns]
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        N1, Ns, N2, // m, n, k
        &alpha,
        w2, N2, // A(k x m)=w3
        delta1, N2, // B(k x n)=delta
        &beta,
        tmp2, N1); // C(m x n)=tmp1 
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSgemm tmp1[N2, Ns] = w3[N3, N2].T * delta[N3, Ns] returned error code %d\n", status);

    // delta2[N1, Ns] = tmp2[N1, Ns] * self.dfactivation(z1)[N1, Ns]
    status = cublasSdgmm(handle, CUBLAS_SIDE_LEFT, 
        N1 * Ns, 1, // m, n
        tmp2, N1 * Ns, // A(m x n)=tmp2
        z1, 1, // X(1 x m)=z1
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

    delete [] Vone_h; cudaFree(Vone_d);
    cudaFree(delta); cudaFree(delta1); cudaFree(delta2);
    cudaFree(tmp1); cudaFree(tmp2);    
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
    const float lr, const float *dw1, const float *dw2, const float *dw3, const float *db1, const float *db2, const float *db3, 
    const int Ns, const int N0, const int N1, const int N2, const int N3){

    cublasStatus_t status;
    cublasHandle_t handle;

    cublasCreate(&handle);

    float alpha = -lr;

    // w1[N1, N0] = w1[N1, N0] - lr * dw1[N1, N0]
    status = cublasSaxpy(handle, N1 * N0, &alpha, dw1, 1, w1, 1);
    if (status != CUBLAS_STATUS_SUCCESS) 
        printf("cublasSaxpy w1[N1, N0] = w1[N1, N0] - lr * dw1[N1, N0] returned error code %d\n", status);

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

}