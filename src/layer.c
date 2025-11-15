#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "layer.h"


void KernelAddLayer(KERNEL *kernel_layer, int n_filter, int row, int col)
{
    kernel_layer->n_filter = n_filter;
    kernel_layer->row = row;
    kernel_layer->col = col;
    

    kernel_layer->kernel = (float *)malloc(sizeof(float) * n_filter * row * col);

    if(NULL == kernel_layer->kernel) {
        printf("커널 층 생성 실패");
        exit(1);
    }

    int fan_in = row * col;
    int fan_out = n_filter;
    int total = fan_in * fan_out;

    he_init(kernel_layer->kernel, fan_in, fan_out, total);
}


void FreeKernel(KERNEL *kernel_layer)
{
    if(NULL != kernel_layer->kernel){
        free(kernel_layer->kernel);
        kernel_layer->kernel = NULL;
    }
}

// 정규분포 난수 생성 by gpt

float normal_rand() {
    float u1 = ((float)rand() + 1) / ((float)RAND_MAX + 2.0f);
    float u2 = ((float)rand() + 1) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}


// He 초기화 by gpt
void he_init(float *weights, int fan_in, int fan_out, int total)
{
    float std = sqrtf(2.0f / fan_in);
    for (int i = 0; i < total; i++)
        weights[i] = std * normal_rand();
    printf("초기화 완료\n");
}

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
