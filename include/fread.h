#ifndef FREAD_H
#define FREAD_H

typedef struct{
    unsigned char *images;
    unsigned char *labels;
    unsigned int num_imgs;
    unsigned int num_labels;
    unsigned int rows;
    unsigned int cols;
    
}MNIST;

int Freader(MNIST *data);

void FreeMNIST(MNIST *data);

void normalize_imgs(MNIST *data, float *out_images);


#endif