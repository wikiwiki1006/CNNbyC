#ifndef LAYER_H
#define LAYER_H

// ====커널 층 생성을 위함====
typedef struct{
    float *kernel;
    int n_filter;
    int row;
    int col;
}KERNEL;

typedef struct{

}CONV;

typedef struct{

}POOL;

void KernelAddLayer(KERNEL *kernel_layers,  int n_filter, int row, int col);

void KernelFree(KERNEL *kernel_layers);

void he_init(float *weights, int fan_in, int fan_out, int total);

void FreeKernel(KERNEL *kernel_layer);
#endif