#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include "layer.h"
#include "calc.h"

// ======convolution forward========
void ConvForward(CONV *conv, double *input, KERNEL *kernel)
{
    int C_in   = kernel->in_c;   // 입력 채널
    int C_out  = kernel->n_filter;  // 출력 채널(필터 수)
    int k      = kernel->k_size;
    int stride = conv->stride;
    int pad    = conv->padding;

    int H_in = conv->in_height;
    int W_in = conv->in_width;

    int H_out = conv->out_cheight;
    int W_out = conv->out_cwidth;

    double *W = kernel->k_weights;   // [C_out][C_in][k][k]
    double *B = kernel->k_biases;     // [C_out]
    double *Y = conv->outputs;      // [C_out][H_out][W_out]

    conv->input = input; // backward용 저장

    // --- Forward ---
    for (int f = 0; f < C_out; f++) {    
        for (int oy = 0; oy < H_out; oy++) {       
            for (int ox = 0; ox < W_out; ox++) {    

                double sum = 0.0;

                // 입력 채널 모두 합산
                for (int c = 0; c < C_in; c++) {
                    for (int ky = 0; ky < k; ky++) {
                        for (int kx = 0; kx < k; kx++) {

                            int iy = oy * stride + ky - pad;
                            int ix = ox * stride + kx - pad;

                            if (iy >= 0 && iy < H_in && ix >= 0 && ix < W_in)
                            {int in_idx = c * (H_in * W_in) + iy * W_in + ix;
                                int w_idx = f * (C_in * k * k) + c * (k * k) + ky * k + kx;
                                sum += input[in_idx] * W[w_idx];
                            }
                        }
                    }
                }

                // bias 추가
                sum += B[f];

                // Output저장
                int out_idx = f * (H_out * W_out) + oy * W_out + ox;
                Y[out_idx] = sum;
            }
        }
    }
    printf("성공!!!!!!!!");
}

// ======활성화 함수를 위한 ReLU함수======
double *ReLU(const double *input, int n_in)
{
    double *out = calloc(n_in, sizeof(double));
    if (out == NULL) return NULL;

    for (int i = 0; i < n_in; i++) {
        out[i] = (input[i] > 0.0 ? input[i] : 0.0);
    }
    printf("ReLU성공!!!!");
    return out; // free(out)해야함
}

// =====마지막 최종 확률을 구하기 위한 SoftMax함수========

double *SoftMax(const double *input, int n_in)
{
    double *prob = malloc(sizeof(double) * n_in);
    if (prob == NULL) return NULL;

    double max = input[0]; // overflow방지를 위해 input에 max값 빼줘야 함
    for (int i = 1; i < n_in; i++) {
        if (input[i] > max) max = input[i];
    }

    // exp(x_i - max) 계산하면서 합 구하기
    double sum = 0.0;
    for (int i = 0; i < n_in; i++) {
        prob[i] = exp(input[i] - max);
        sum += prob[i];
    }

    for (int i = 0; i < n_in; i++) {
        prob[i] /= sum;
    }

    return prob;   // free(prob) 해야 함
}
// =====배치별 평균 Binary Cross Entropy손실 함수======
/* 호출 하기 전 y_true값을 double 형인지 확인해야함*/

double BCE(const int *y_true, const double *y_pred, int n)
{
    const double eps = 1e-7;   // 0, 1에 너무 가까울 때 log 커지는 상황 방지
    double loss_sum = 0.0;

    for (int i = 0; i < n; i++) {
        double p = y_pred[i];

        // 범위 클리핑
        if (p < eps) p = eps;
        else if (p > 1.0 - eps) p = 1.0 - eps;

        loss_sum += -( y_true[i] * log(p) +
                       (1.0 - y_true[i]) * log(1.0 - p) );
    }

    // 배치 평균 loss
    return loss_sum / n;
}

