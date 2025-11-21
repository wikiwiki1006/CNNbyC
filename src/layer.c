#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stddef.h>
#include "layer.h"
#include <assert.h>



// =======Convolution층과 FC층 연결 노드 생성============
static FLAYER* AddFlattenLayer(int n_filter, int width, int height, int nnodes)
{
    FLAYER* self = (FLAYER*)calloc(1, sizeof(FLAYER));
    if (self == NULL) return NULL;

    self->n_filter = n_filter;
    self->width = width;
    self->height = height;
    
    /* Nnodes: number of outputs. */
    self->nnodes = nnodes;
    self->outputs = (double*)calloc(self->nnodes, sizeof(double));
    self->gradients = (double*)calloc(self->nnodes, sizeof(double));
    self->errors = (double*)calloc(self->nnodes, sizeof(double));

    self->nbiases = nnodes;
    self->biases = (double*)calloc(self->nbiases, sizeof(double));

    self->nweights = n_filter * width * height * nnodes;
    self->weights = (double*)calloc(self->nweights, sizeof(double));

    return self;
}

/* Layer_destroy(self)
   Releases the memory.
*/
void Layer_destroy(FLAYER* self)
{
    assert (self != NULL);

    free(self->outputs);
    free(self->gradients);
    free(self->errors);

    free(self->biases);
    free(self->weights);

    free(self);
}

// ============완전 연결 층 생성============

static FCLAYER* AddFCLayer(int nnodes, int prev_nnodes)
{
    {
    FCLAYER* self = (FCLAYER*)calloc(1, sizeof(FCLAYER));
    if (self == NULL) return NULL;

    /* Nnodes: number of outputs. */
    self->outputs = (double*)calloc(nnodes, sizeof(double));
    self->gradients = (double*)calloc(nnodes, sizeof(double));
    self->errors = (double*)calloc(nnodes, sizeof(double));

    self->nbiases = nnodes;
    self->biases = (double*)calloc(self->nbiases, sizeof(double));

    self->nweights = prev_nnodes * nnodes;
    self->weights = (double*)calloc(self->nweights, sizeof(double));

    return self;
}
}


// =============kernel층 생성 및 초기화===============

static KERNEL* AddKernelLayer(int n_filter, int k_size)
{
    // 1) 구조체 자체 할당
    KERNEL *self = calloc(1, sizeof(KERNEL));
    if (self == NULL) {
        fprintf(stderr, "KERNEL 구조체 할당 실패\n");
        exit(1);
    }

    self->n_filter = n_filter;
    self->k_size   = k_size;

    // 2) weight / bias 크기 계산
    int fan_in_per_filter = k_size * k_size;               // 입력 채널 1개 가정
    int total_weights     = n_filter * fan_in_per_filter;  // 전체 weight 개수

    // 3) 메모리 할당 (calloc 인자 순서 주의)
    self->k_weight = calloc(total_weights, sizeof(double));
    self->k_bias   = calloc(n_filter,      sizeof(double));

    if (self->k_weight == NULL || self->k_bias == NULL) {
        fprintf(stderr, "커널 weight/bias 메모리 할당 실패\n");
        free(self->k_weight);
        free(self->k_bias);
        free(self);
        exit(1);
    }

    // 4) He 초기화 (fan_in = 필터당 weight 수)
    kernel_he_init(self->k_weight, k_size, n_filter);

    // bias는 보통 0으로 초기화
    for (int i = 0; i < n_filter; i++) {
        self->k_bias[i] = 0.0;
    }

    printf("커널 층 생성 및 초기화 완료\n");
    return self;
}


void FreeKernel(KERNEL *kernel_layer)
{
    if((NULL != kernel_layer->k_weight) && (NULL != kernel_layer->k_bias)){
        free(kernel_layer->k_weight);
        free(kernel_layer->k_bias);
        kernel_layer->k_weight = NULL;
        kernel_layer->k_bias = NULL;
    }
}
//==================================

// =======convolution 층 생성=========
static CONV* AddConv(int n_filter, int in_width, int in_height, int krow, int kcol, int padding, int stride)
{
    CONV* self = (CONV*) calloc(1, sizeof(CONV));
    self->out_cheight = ((in_height + 2 * padding - krow) / stride) + 1;
    self->out_cwidth = ((in_width + 2 * padding - kcol) / stride) + 1;

    self->outputs = (double*)calloc(n_filter * in_width * in_height, sizeof(double));
    self->gradients = (double*)calloc(n_filter * in_width * in_height, sizeof(double));

    return self;
    
}

void FreeConv(CONV *conv){
    if(conv->outputs == NULL && conv->gradients == NULL) return;
    free(conv->outputs);
    free(conv->gradients);
    conv->outputs = NULL;
    conv->gradients = NULL;

    conv->out_cheight = 0;
    conv->out_cwidth = 0;
}
// ================================

//===========pooling층 생성=============
static POOL* AddPool(int in_channel, int in_width, int in_height, int prow, int pcol, int padding, int stride)
{
    POOL* self = (POOL*) calloc(1, sizeof(POOL));
    self->out_pheight = ((in_height + 2 * padding - prow) / stride) + 1;
    self->out_pwidth = ((in_width + 2 * padding - pcol) / stride) + 1;

    self->lpool = (double *)calloc(in_channel * self->out_pwidth * self->out_pheight, sizeof(double));

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


// TODO input과 kenel을 계산해서 conv층으로 만들어주는 함수 구현

// TODO pooling층 계산하는 함수 구현

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
    printf("초기화 완료\n");
}

void kernel_he_init(double *weights, int kernel_size, int n_filter)
{
    int fan_in = kernel_size * kernel_size;        // He에서 사용할 fan_in
    double std = sqrt(2.0 / (double)fan_in);

    for (int i = 0; i < n_filter; i++) {
        for (int j = 0; j < fan_in; j++) {
            int idx = i * fan_in + j;
            weights[idx] = std * normal_rand();
        }
    }
    printf("초기화 완료\n");
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
