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
 * @brief perform neural network with gradient descent algorithm
 * 
 * @param lr[in] learning rate
 * @param epoch[in] max number of epoch
 * @param accuracy[in] required accuracy
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
void compute_nn(float lr, int epoch, float accuracy, float *w1, float *w2, float *w3, float *b1, float *b2, float *b3, 
    float *X, float *Y, float *Y_h, int Ns, int N0, int N1, int N2, int N3){

    // Pointer to device arrays
    float *a1, *a2, *a3;
    float *z1, *z2, *z3;
    float *dw1, *dw2, *dw3;
    float *db1, *db2, *db3;

    float *delta, *delta1, *delta2;
    float *tmp1, *tmp2, *dz1, *dz2;    

    float *a3_h  = new float[N3 * Ns];
    double *a3_hd = new double[N3 * Ns];    

    float acc;
    float *Vone_h = new float[Ns];
    float *Vone_d;
    size_t size_Vone = Ns * sizeof(float);

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

    cudaMalloc((void **) &delta,  size_a3);
    cudaMalloc((void **) &delta1, size_a2); cudaMalloc((void **) &tmp1, size_a2); cudaMalloc((void **) &dz1, size_a2);
    cudaMalloc((void **) &delta2, size_a1); cudaMalloc((void **) &tmp2, size_a1); cudaMalloc((void **) &dz2, size_a1);

    // Vone on host
    for (int i = 0; i < Ns; i++)
        Vone_h[i] = 1.0f;
    
    // Vone on device
    cudaMalloc((void **) &Vone_d, size_Vone);
    cudaMemcpy(Vone_d, Vone_h, size_Vone, cudaMemcpyHostToDevice);

    for (int i = 0; i < epoch; i++){

        // if(i*10%epoch == 0) printf("===");
        
        // forward propagationx
        forward_propagation(a1, a2, a3, z1, z2, z3, 
            w1, w2, w3, b1, b2, b3, 
            X, Vone_d, a3_h, a3_hd, 
            Ns, N0, N1, N2, N3);

        // backward propagation
        backward_propagation(dw1, dw2, dw3, db1, db2, db3,
            w1, w2, w3,
            a1, a2, a3, z1, z2, z3,
            X, Y, Vone_d,
            delta, delta1, delta2, 
            tmp1, tmp2, dz1, dz2,
            Ns, N0, N1, N2, N3);
            
        // update weights and biases
        update_parameter(w1, w2, w3, b1, b2, b3,
            lr, dw1, dw2, dw3, db1, db2, db3, 
            Ns, N0, N1, N2, N3);

        acc = get_accuracy(a3_h, Y_h, N3, Ns);
        if (acc > accuracy){
            printf("i=%d, accuracy=%.3f\n", i, acc);
            break;
        }
        if (i % 100 == 0){
            acc = get_accuracy(a3_h, Y_h, N3, Ns);
            printf("i=%d, accuracy=%.3f\n", i, acc);
        }
    }

    forward_propagation(a1, a2, a3, z1, z2, z3, 
        w1, w2, w3, b1, b2, b3, 
        X, Vone_d, a3_h, a3_hd,
        Ns, N0, N1, N2, N3);
    printf("accuracy=%.3f\n", get_accuracy(a3_h, Y_h, N3, Ns));

    // free memory on device
    cudaFree(a1); cudaFree(a2); cudaFree(a3);
    cudaFree(z1); cudaFree(z2); cudaFree(z3);
    cudaFree(dw1); cudaFree(dw2); cudaFree(dw3);
    cudaFree(db1); cudaFree(db2); cudaFree(db3);
    delete [] Vone_h; cudaFree(Vone_d);
    delete [] a3_h; delete [] a3_hd;
    cudaFree(delta); cudaFree(delta1); cudaFree(delta2);
    cudaFree(tmp1); cudaFree(tmp2);
    cudaFree(dz1); cudaFree(dz2);
}

/**
 * @brief perform neural network with ADAM algorithm
 * 
 * @param lr[in] learning rate
 * @param epoch[in] max number of epoch
 * @param accuracy[in] required accuracy
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
void compute_nn_adam(float lr, int epoch, float accuracy, float *w1, float *w2, float *w3, float *b1, float *b2, float *b3, 
    float *X, float *Y, float *Y_h, int Ns, int N0, int N1, int N2, int N3){

     // ADAM parameter
    float beta1 = 0.9;
    float beta2 = 0.999;

    // Pointer to device arrays
    float *a1, *a2, *a3;
    float *z1, *z2, *z3;
    float *dw1, *dw2, *dw3;
    float *db1, *db2, *db3;
    float *dw21, *dw22, *dw23;
    float *db21, *db22, *db23;
    float *mw1, *mw2, *mw3;
    float *mb1, *mb2, *mb3;
    float *vw1, *vw2, *vw3;
    float *vb1, *vb2, *vb3;
    float *mhatw1, *mhatw2, *mhatw3;
    float *mhatb1, *mhatb2, *mhatb3;
    float *vhatw1, *vhatw2, *vhatw3;
    float *vhatb1, *vhatb2, *vhatb3;

    float *delta, *delta1, *delta2;
    float *tmp1, *tmp2, *dz1, *dz2;    

    float *a3_h  = new float[N3 * Ns];
    double *a3_hd = new double[N3 * Ns];    

    float acc;
    float *Vone_h = new float[Ns];
    float *Vone_d;
    size_t size_Vone = Ns * sizeof(float);

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

    cudaMalloc((void **) &delta,  size_a3);
    cudaMalloc((void **) &delta1, size_a2); cudaMalloc((void **) &tmp1, size_a2); cudaMalloc((void **) &dz1, size_a2);
    cudaMalloc((void **) &delta2, size_a1); cudaMalloc((void **) &tmp2, size_a1); cudaMalloc((void **) &dz2, size_a1);

    cudaMalloc((void **) &dw21, N1 * N0 * sizeof(float));
    cudaMalloc((void **) &dw22, N2 * N1 * sizeof(float));
    cudaMalloc((void **) &dw23, N3 * N2 * sizeof(float));

    cudaMalloc((void **) &db21, N1 * sizeof(float));
    cudaMalloc((void **) &db22, N2 * sizeof(float));
    cudaMalloc((void **) &db23, N3 * sizeof(float));

    cudaMalloc((void **) &mw1, N1 * N0 * sizeof(float));
    cudaMalloc((void **) &mw2, N2 * N1 * sizeof(float));
    cudaMalloc((void **) &mw3, N3 * N2 * sizeof(float));

    cudaMalloc((void **) &mb1, N1 * sizeof(float));
    cudaMalloc((void **) &mb2, N2 * sizeof(float));
    cudaMalloc((void **) &mb3, N3 * sizeof(float));

    cudaMalloc((void **) &vw1, N1 * N0 * sizeof(float));
    cudaMalloc((void **) &vw2, N2 * N1 * sizeof(float));
    cudaMalloc((void **) &vw3, N3 * N2 * sizeof(float));

    cudaMalloc((void **) &vb1, N1 * sizeof(float));
    cudaMalloc((void **) &vb2, N2 * sizeof(float));
    cudaMalloc((void **) &vb3, N3 * sizeof(float));

    cudaMalloc((void **) &mhatw1, N1 * N0 * sizeof(float));
    cudaMalloc((void **) &mhatw2, N2 * N1 * sizeof(float));
    cudaMalloc((void **) &mhatw3, N3 * N2 * sizeof(float));

    cudaMalloc((void **) &mhatb1, N1 * sizeof(float));
    cudaMalloc((void **) &mhatb2, N2 * sizeof(float));
    cudaMalloc((void **) &mhatb3, N3 * sizeof(float));

    cudaMalloc((void **) &vhatw1, N1 * N0 * sizeof(float));
    cudaMalloc((void **) &vhatw2, N2 * N1 * sizeof(float));
    cudaMalloc((void **) &vhatw3, N3 * N2 * sizeof(float));

    cudaMalloc((void **) &vhatb1, N1 * sizeof(float));
    cudaMalloc((void **) &vhatb2, N2 * sizeof(float));
    cudaMalloc((void **) &vhatb3, N3 * sizeof(float));

    // Vone on host
    for (int i = 0; i < Ns; i++)
        Vone_h[i] = 1.0f;
    
    // Vone on device
    cudaMalloc((void **) &Vone_d, size_Vone);
    cudaMemcpy(Vone_d, Vone_h, size_Vone, cudaMemcpyHostToDevice);

    for (int i = 0; i < epoch; i++){

        // if(i*10%epoch == 0) printf("===");
        
        // forward propagationx
        forward_propagation(a1, a2, a3, z1, z2, z3, 
            w1, w2, w3, b1, b2, b3, 
            X, Vone_d, a3_h, a3_hd, 
            Ns, N0, N1, N2, N3);

        // backward propagation
        backward_propagation(dw1, dw2, dw3, db1, db2, db3,
            w1, w2, w3,
            a1, a2, a3, z1, z2, z3,
            X, Y, Vone_d,
            delta, delta1, delta2, 
            tmp1, tmp2, dz1, dz2,
            Ns, N0, N1, N2, N3);
            
        // update weights and biases
        update_parameter_adam(w1, w2, w3, b1, b2, b3,
            i+1, lr, beta1, beta2,
            dw1, dw2, dw3, db1, db2, db3,
            dw21, dw22, dw23, db21, db22, db23,
            mw1, mw2, mw3, mb1, mb2, mb3,
            vw1, vw2, vw3, vb1, vb2, vb3,
            mhatw1, mhatw2, mhatw3, mhatb1, mhatb2, mhatb3,
            vhatw1, vhatw2, vhatw3, vhatb1, vhatb2, vhatb3,
            Ns, N0, N1, N2, N3);

        acc = get_accuracy(a3_h, Y_h, N3, Ns);
        if (acc > accuracy){
            printf("i=%d, accuracy=%.3f\n", i, acc);
            break;
        }
        if (i % 100 == 0){
            acc = get_accuracy(a3_h, Y_h, N3, Ns);
            printf("i=%d, accuracy=%.3f\n", i, acc);
        }
    }

    forward_propagation(a1, a2, a3, z1, z2, z3, 
        w1, w2, w3, b1, b2, b3, 
        X, Vone_d, a3_h, a3_hd,
        Ns, N0, N1, N2, N3);
    printf("accuracy=%.3f\n", get_accuracy(a3_h, Y_h, N3, Ns));

    // free memory on device
    cudaFree(a1); cudaFree(a2); cudaFree(a3);
    cudaFree(z1); cudaFree(z2); cudaFree(z3);
    cudaFree(dw1); cudaFree(dw2); cudaFree(dw3);
    cudaFree(db1); cudaFree(db2); cudaFree(db3);
    delete [] Vone_h; cudaFree(Vone_d);
    delete [] a3_h; delete [] a3_hd;
    cudaFree(delta); cudaFree(delta1); cudaFree(delta2);
    cudaFree(tmp1); cudaFree(tmp2);
    cudaFree(dz1); cudaFree(dz2);
    cudaFree(dw21); cudaFree(dw22); cudaFree(dw23);
    cudaFree(db21); cudaFree(db22); cudaFree(db23);
    cudaFree(mw1); cudaFree(mw2); cudaFree(mw3);
    cudaFree(mb1); cudaFree(mb2); cudaFree(mb3);
    cudaFree(vw1); cudaFree(vw2); cudaFree(vw3);
    cudaFree(vb1); cudaFree(vb2); cudaFree(vb3);
    cudaFree(mhatw1); cudaFree(mhatw2); cudaFree(mhatw3);
    cudaFree(mhatb1); cudaFree(mhatb2); cudaFree(mhatb3);
    cudaFree(vhatw1); cudaFree(vhatw2); cudaFree(vhatw3);
    cudaFree(vhatb1); cudaFree(vhatb2); cudaFree(vhatb3);
}