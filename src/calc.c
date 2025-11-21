#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include "calc.h"


// ======활성화 함수를 위한 ReLU함수======
double *ReLU(const double *input, int n_in)
{
    double *out = calloc(n_in, sizeof(double));
    if (out == NULL) return NULL;

    for (int i = 0; i < n_in; i++) {
        out[i] = (input[i] > 0.0 ? input[i] : 0.0);
    }
    return out; // free(out)해야함
}

// =====마지막 최종 확률을 구하기 위한 SoftMax함수========
#include <stdlib.h>
#include <math.h>

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

#include <math.h>
#include <stdlib.h>

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

