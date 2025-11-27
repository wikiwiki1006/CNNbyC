#define _USE_MATH_DEFINES

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include "layer.h"
#include <assert.h>



// =======Convolution층과 FC층 연결 노드 생성============
FLLAYER* AddFlattenLayer(int in_channel, int in_width, int in_height)
{
    FLLAYER* self = (FLLAYER*)calloc(1, sizeof(FLLAYER));
    if (self == NULL) return NULL;

    self->in_channel = in_channel;
    self->nnodes = in_channel * in_width * in_height;
    self->outputs = (double*)calloc(self->nnodes, sizeof(double));
	self->delta = (double*)calloc(self->nnodes, sizeof(double));

    return self;
    }

void Layer_destroy(FLLAYER* self)
{
    assert (self != NULL);

    free(self->outputs);
    free(self);
}

// ============완전 연결 층 생성============

FCLAYER* AddFCLayer(int nnodes, int prev_nnodes)
{
    {
    FCLAYER* self = (FCLAYER*)calloc(1, sizeof(FCLAYER));
    if (self == NULL) return NULL;

    self->nweights = prev_nnodes * nnodes;
    self->nbiases = nnodes;
    self->nnodes = nnodes;

    self->outputs = (double*)calloc(nnodes, sizeof(double));
    self->z = (double*)calloc(nnodes, sizeof(double));
    self->dweights = (double*)calloc(self->nweights, sizeof(double));
    self->dbiases = (double*)calloc(self->nbiases, sizeof(double));
    self->delta = (double*)calloc(self->nbiases, sizeof(double));
    self->biases = (double*)calloc(self->nbiases, sizeof(double));
    self->weights = (double*)calloc(self->nweights, sizeof(double));

    layer_he_init(self->weights, prev_nnodes, self->nnodes);

    return self;
    
    }
}


// =============kernel층 생성 및 초기화===============

KERNEL* AddKernelLayer(int in_c, int in_w, int in_h, int n_filter, int k_size)
{
    KERNEL *self = calloc(1, sizeof(KERNEL));
    if (self == NULL) {
        fprintf(stderr, "KERNEL 구조체 할당 실패\n");
        exit(1);
    }

    self->n_filter = n_filter;
    self->k_size   = k_size;
    self->in_c = in_c;
    int fan_in = in_c * k_size * k_size;
    int total_weights  = in_c * n_filter * k_size * k_size;  // 전체 weight 개수

	self->nweights = total_weights;
	self->nbiases = n_filter;

    // 메모리 할당
    self->k_weights = calloc(total_weights, sizeof(double));
    self->k_biases   = calloc(n_filter, sizeof(double));
    self->dweights = calloc(self->nweights, sizeof(double));
    self->dbiases = calloc(self->nbiases, sizeof(double));


    if (self->k_weights == NULL || self->k_biases == NULL) {
        fprintf(stderr, "커널 weight/bias 메모리 할당 실패\n");
        free(self->k_weights);
        free(self->k_biases);
        free(self);
        exit(1);
    }

    // He 초기화 (fan_in = 필터당 weight 수)
    kernel_he_init(self->k_weights, fan_in, n_filter);

    // bias는 보통 0으로 초기화
    for (int i = 0; i < n_filter; i++) {
        self->k_biases[i] = 0.0;
    }

    return self;
}


void FreeKernel(KERNEL *kernel_layer)
{
    if((NULL != kernel_layer->k_weights) && (NULL != kernel_layer->k_biases)){
        free(kernel_layer->k_weights);
        free(kernel_layer->k_biases);
        kernel_layer->k_weights = NULL;
        kernel_layer->k_biases = NULL;
    }
}
//==================================

// =======convolution 층 생성=========
CONV* AddConv(int n_filter, int in_width, int in_height, int k_size, int padding, int stride)
{
    CONV* self = (CONV*) calloc(1, sizeof(CONV));
    self->padding = padding;
    self->stride = stride;
    self->out_cheight = ((in_height + 2 * padding - k_size) / stride) + 1;
    self->out_cwidth = ((in_width + 2 * padding - k_size) / stride) + 1;

    self->channel = n_filter;
    self->in_width = in_width;
    self->in_height = in_height;

    self->outputs = (double*)calloc(n_filter * self->out_cwidth * self->out_cheight, sizeof(double));
    self->z = (double*)calloc(n_filter * self->out_cwidth * self->out_cheight, sizeof(double));
    self->delta = (double*)calloc(n_filter * self->out_cwidth * self->out_cheight, sizeof(double));

    return self;
    
    
    }

void FreeConv(CONV *conv){
    if(conv->outputs == NULL && conv->delta == NULL) return;
    free(conv->outputs);
    free(conv->delta);
    conv->outputs = NULL;
    conv->delta = NULL;

    conv->out_cheight = 0;
    conv->out_cwidth = 0;
}
// ================================

//===========pooling층 생성=============
POOL* AddPool(int in_channel, int in_width, int in_height, int prow, int pcol, int padding, int stride)
{
    POOL* self = (POOL*) calloc(1, sizeof(POOL));
    self->prow = prow;
    self->pcol = pcol;
    self->padding = padding;
    self->stride = stride;
    self->out_pheight = ((in_height + 2 * padding - prow) / stride) + 1;
    self->out_pwidth = ((in_width + 2 * padding - pcol) / stride) + 1;
    self->channel = in_channel;

    self->maxidx = (int *)calloc(self->channel * self->out_pwidth * self->out_pheight, sizeof(int));
    self->lpool = (double *)calloc(self->channel * self->out_pwidth * self->out_pheight, sizeof(double));
    self->delta = (double *)calloc(self->channel * self->out_pwidth * self->out_pheight, sizeof(double));

    return self;
}

void FreePool(POOL *pool){
    if(pool->lpool == NULL) return;
    if(pool->lpool != NULL){
        free(pool->lpool);
        pool->lpool = NULL;
    }
    pool->out_pheight = 0;
    pool->out_pwidth = 0;
}
// ==================================


// TODO input과 kenel을 계산해서 conv층으로 만들어주는 함수 구현 -> 완료

// TODO pooling층 계산하는 함수 구현 -> 완료

// TODO FC층 생성 함수 구현 -> 완료

// ============ 랜덤 초기화==============

// 정규분포 난수 생성 by gpt(너무 수학적입니다)

double normal_rand(void) {
    double u1 = ((double)rand() + 1) / ((double)RAND_MAX + 2.0f);
    double u2 = ((double)rand() + 1) / ((double)RAND_MAX + 2.0f);
    return sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);
}

// He 초기화 by gpt(너무 수학적입니다)
void layer_he_init(double *weights, int fan_in, int fan_out)
{
    double total = fan_in * fan_out;
    double std = sqrt(2.0f / fan_in);
    for (int i = 0; i < total; i++)
        weights[i] = std * normal_rand();
}

void kernel_he_init(double *weights, int fan_in, int n_filter)
{
    double std = sqrt(2.0 / (double)fan_in);

    for (int i = 0; i < n_filter; i++) {
        for (int j = 0; j < fan_in; j++) {
            int idx = i * fan_in + j;
            weights[idx] = std * normal_rand();
        }
    }
}
// ===================================


#ifdef DEBUG
int main(void)
{
    KERNEL k;
    KernelLayer(&k, 4, 3, 3);  // 4개의 3x3 필터 생성
    printf("필터 개수 : %d, 행 개수 : %d, 열 개수 : %d", k.n_filter, k.row, k.col);
    FreeKernel(&k);
    return 0;
}
#endif
