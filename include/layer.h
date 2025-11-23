#ifndef LAYER_H
#define LAYER_H

/*  Layer
 */
typedef struct _FLLayer {

    int in_width;
    int in_height;
    int in_channel;
    int nnodes;   

    double* delta;
    double* outputs;  

} FLLAYER;

typedef struct _FCLayer {
    int nnodes;
    int nbiases;
    int nweights;

    double* biases;         
    double* weights;    

    double* outputs; 
    double* z;     
    double* delta;     
    double* dweights;
    double* dbiases; 


} FCLAYER;

// 커널 층 생성
typedef struct{
    double *k_weights;
    double *k_biases;

    double* dweights;
    double *dbiases;

    int n_filter;
    int k_size;
    int in_c;
} KERNEL;

// 커널 통과한 합성곱 층 생성
typedef struct{
    double *outputs; //출력 값 정보
    double *gradients; //역전파 계산시 grad
    double *input;
    double *z;

    int channel;
    int out_cwidth;
    int out_cheight;
    int in_width;
    int in_height;
    int stride;
    int padding;
}CONV;

//max pooling층 생성
typedef struct{
    double *lpool;
    double *delta;

    int *maxidx;
    int pcol;
    int prow;
    int padding;
    int stride;
    int out_pwidth;
    int out_pheight;
    int channel;

}POOL;

FLLAYER* AddFlattenLayer(int n_filter, int in_width, int in_height);

void Layer_destroy(FLLAYER* self);

FCLAYER* AddFCLayer(int nnodes, int prev_nnodes);

void FreeFCLayer(FCLAYER);

KERNEL* AddKernelLayer(int in_c, int in_w, int in_h, int n_filter, int k_size);

void FreeKernel(KERNEL *kernel_layer);

void layer_he_init(double *weights, int fan_in, int fan_out);

void kernel_he_init(double *k_weight, int k_size, int n_filter);

CONV* AddConv(int n_filter, int in_width, int in_height, int k_size, int padding, int stride);

void FreeConv(CONV *conv);

POOL* AddPool(int in_channel, int in_width, int in_height, int prow, int pcol, int padding, int stride);

void FreePool(POOL *pool);


#endif