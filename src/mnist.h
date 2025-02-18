/*
Takafumi Hoiruchi. 2018.
https://github.com/takafumihoriuchi/MNIST_for_C
*/

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include "io.h"

// set appropriate path for data
// #define TRAIN_IMAGE "./data/train-images.idx3-ubyte"
// #define TRAIN_LABEL "./data/train-labels.idx1-ubyte"
// #define TEST_IMAGE "./data/t10k-images.idx3-ubyte"
// #define TEST_LABEL "./data/t10k-labels.idx1-ubyte"

#define SIZE 784 // 28*28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
// #define LEN_INFO_IMAGE 4
// #define LEN_INFO_LABEL 2

// #define MAX_IMAGESIZE 1280
// #define MAX_BRIGHTNESS 255
// #define MAX_FILENAME 256
// #define MAX_NUM_OF_IMAGES 1

// unsigned char image[MAX_NUM_OF_IMAGES][MAX_IMAGESIZE][MAX_IMAGESIZE];
// int width[MAX_NUM_OF_IMAGES], height[MAX_NUM_OF_IMAGES];

// int info_image[LEN_INFO_IMAGE];
// int info_label[LEN_INFO_LABEL];

unsigned char train_image_char[NUM_TRAIN][SIZE];
unsigned char test_image_char[NUM_TEST][SIZE];
unsigned char train_label_char[NUM_TRAIN][1];
unsigned char test_label_char[NUM_TEST][1];

double train_image[NUM_TRAIN][SIZE];
double test_image[NUM_TEST][SIZE];
int  train_label[NUM_TRAIN];
int test_label[NUM_TEST];


void FlipLong(unsigned char * ptr)
{
    unsigned char val;
    
    // Swap 1st and 4th bytes
    val = *(ptr);
    *(ptr) = *(ptr+3);
    *(ptr+3) = val;
    
    // Swap 2nd and 3rd bytes
    ptr += 1;
    val = *(ptr);
    *(ptr) = *(ptr+1);
    *(ptr+1) = val;
}

template<size_t M, size_t N>
void read_mnist_char(char *file_path, int num_data, int len_info, int arr_n, unsigned char (&data_char)[M][N], int info_arr[])
{
    int i, fd;
    unsigned char *ptr;

    if ((fd = _open(file_path, O_RDONLY)) == -1) {
        fprintf(stderr, "couldn't open image file");
        exit(-1);
    }
    
    _read(fd, info_arr, len_info * sizeof(int));
    
    // read-in information about size of data
    for (i=0; i<len_info; i++) { 
        ptr = (unsigned char *)(info_arr + i);
        FlipLong(ptr);
        ptr = ptr + sizeof(int);
    }
    printf("num_data=%d, len_info=%d, arr_n=%d\n", num_data, len_info, arr_n);
    for (i = 0; i < len_info; i++)
        printf("(%d) %d\n", i, info_arr[i]);
    
    // read-in mnist numbers (pixels|labels)
    for (i=0; i<num_data; i++) {
        _read(fd, data_char[i], arr_n * sizeof(unsigned char));
    }

    // if (arr_n == 784){
    //     for (i=0; i<arr_n; i++) {
    //     printf("%.1f ", (double)(data_char[100][i]/255.0f));
    //         if ((i+1) % 28 == 0) putchar('\n');
    //     }
    // }
    // printf("\n");

    _close(fd);
}


void image_char2double(int num_data, unsigned char data_image_char[][SIZE], double data_image[][SIZE])
{
    int i, j;
    for (i=0; i<num_data; i++)
        for (j=0; j<SIZE; j++)
            data_image[i][j]  = (double)data_image_char[i][j] / 255.0;
}


void label_char2int(int num_data, unsigned char data_label_char[][1], int data_label[])
{
    int i;
    for (i=0; i<num_data; i++)
        data_label[i]  = (int)data_label_char[i][0];
}

void load_mnist(){

    const int n_INFO_IMAGE = 4;
    const int n_INFO_LABEL = 2;

    int info_image[n_INFO_IMAGE];
    int info_label[n_INFO_LABEL];

    printf("load mnist\n");
    read_mnist_char("data/train-images.idx3-ubyte", NUM_TRAIN, n_INFO_IMAGE, SIZE, train_image_char, info_image);
    image_char2double(NUM_TRAIN, train_image_char, train_image);    

    read_mnist_char("data/t10k-images.idx3-ubyte", NUM_TEST, n_INFO_IMAGE, SIZE, test_image_char, info_image);
    image_char2double(NUM_TEST, test_image_char, test_image);
    
    read_mnist_char("data/train-labels.idx1-ubyte", NUM_TRAIN, n_INFO_LABEL, 1, train_label_char, info_label);
    label_char2int(NUM_TRAIN, train_label_char, train_label);
    
    read_mnist_char("data/t10k-labels.idx1-ubyte", NUM_TEST, n_INFO_LABEL, 1, test_label_char, info_label);
    label_char2int(NUM_TEST, test_label_char, test_label);

}