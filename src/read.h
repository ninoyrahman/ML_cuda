/**
 * @file read.h
 * @author ninoy rahman
 * @brief 
 * @version 0.1
 * @date 2025-02-18
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <iostream>
#include <fstream>

/**
 * @brief read data from text file 
 * 
 * @param filename[in] name of file 
 * @param X[out] data 
 * @param nfeat[in] feature number 
 * @param nsample[in] sample number
 */
void read_image(const char *filename, int *X, const int nfeat, const int nsample){
    std::ifstream myfile (filename, std::ifstream::in);

    if(!myfile){
        std::cout << "File does not exist." << std::endl;
    } else {
        std::cout << "filename= " << filename << std::endl;
    }

    for(int i = 0; i < nsample; i++){
        for(int j = 0; j < nfeat; j++ ){
           myfile >> X[i * nfeat + j];
        }
        if((i+1)*10%nsample == 0) printf("===");
    }    
    printf("\n");

    myfile.close();
}

/**
 * @brief read train and test data from text file
 * 
 * @param X_train[out] features for training
 * @param X_test[out] features for testing 
 * @param Y_test[out] label for training
 * @param Y_train[out] label for testing 
 * @param N0[in] feature number 
 * @param Ntest[in] testing sample size 
 * @param Ntrain[in] training sample size
 */
void read_mnist(float *X_h, float *Y_h, float *X1_h, float *Y1_h, 
    int *X_train, int *X_test, int *Y_test, int *Y_train, 
    const int nfeat, const int nlabel, const int Ntrain, const int Ntest){
    read_image("data/x_train.txt", X_train, nfeat, Ntrain);
    read_image("data/y_train.txt", Y_train, 1, Ntrain);

    read_image("data/x_test.txt", X_test, nfeat, Ntest);
    read_image("data/y_test.txt", Y_test, 1, Ntest);

    // X[N0, Ns], x_train[N0, Ns]
    // Y[N3, Ns], y_train[Ns]
    for (int i = 0; i < Ntrain; i++){
        for (int j = 0; j < nfeat; j++){
            X_h[i * nfeat + j] = (float)(X_train[i * nfeat + j] / 255.0f);
        }
        
        for (int j = 0; j < nlabel; j++){
            Y_h[i * nlabel + j] = 0.0f;
            if (j == Y_train[i]) Y_h[i * nlabel + j] = 1.0f;
        }
    } 
    
    for (int i = 0; i < Ntest; i++){
        for (int j = 0; j < nfeat; j++){
            X1_h[i * nfeat + j] = (float)(X_test[i * nfeat + j] / 255.0f);
        }
        
        for (int j = 0; j < nlabel; j++){
            Y1_h[i * nlabel + j] = 0.0f;
            if (j == Y_test[i]) Y1_h[i * nlabel + j] = 1.0f;
        }
    }     
}