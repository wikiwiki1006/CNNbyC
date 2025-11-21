#ifndef LAYER_H
#define LAYER_H

/*  Layer
 */
typedef struct _FLayer {
    
    int width;
    int height;
    int n_filter;
    int nbiases;
    int nweights;
    int nnodes;   

    double* outputs;    
    double* gradients;    
    double* errors;      
        
    double* biases;         

    double* weights;           

} FLAYER;

typedef struct _FCLayer {
    int nnodes;
    int nbiases;
    int nweights;

    double* outputs;    
    double* gradients;    
    double* errors;      
          
    double* biases;         
    
    double* weights;    



} FCLAYER;

// 커널 층 생성
typedef struct{
    double *k_weight;
    double *k_bias;

    double* gradients;

    int n_filter;
    int k_size;
} KERNEL;

// 커널 통과한 합성곱 층 생성
typedef struct{
    double *outputs; //출력 값 정보
    double *gradients; //역전파 계산시 grad
    int out_cwidth;
    int out_cheight;
}CONV;

//max pooling층 생성
typedef struct{
    double *lpool;
    int out_pwidth;
    int out_pheight;

}POOL;

static FLAYER* AddFlattenLayer(int n_filter, int width, int height, int nnodes);

void Layer_destroy(FLAYER* self);

static FCLAYER* AddFCLayer(int nnodes, int prev_nnodes);

void FreeFCLayer(FCLAYER);

static FCLAYER* AddFCLayer(int nnodes, int prev_nnodes);

void FreeKernel(KERNEL *kernel_layer);

void he_init(double *weights, int fan_in, int fan_out, int total);

void kernel_he_init(double *k_weight, int k_size, int n_filter);

static CONV* AddConv(int n_filter, int in_width, int in_height, int krow, int kcol, int padding, int stride);

void FreeConv(CONV *conv);

static POOL* AddPool(int in_channel, int in_width, int in_height, int prow, int pcol, int padding, int stride);

void FreePool(POOL *pool);


#endif