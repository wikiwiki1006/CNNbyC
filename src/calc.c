#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include <float.h>
#include <assert.h>
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
    printf("ConvForward성공!!!!!!!!\n");
}
//=============풀링 층 forward=============
void PoolForward(POOL *pool, CONV *conv)
{
    int in_c = conv->channel;
    int in_w = conv->out_cwidth;
    int in_h = conv->out_cheight;
    
    int o_w = pool->out_pwidth;
    int o_h = pool->out_pheight;
    int pcol = pool->pcol;
    int prow = pool->prow;
    
    int str = pool->stride;
    int pad = pool->padding;

    double *input = conv->z;

    for(int ch = 0; ch < in_c; ch++){
        for(int w = 0; w < o_w; w++){
            for(int h = 0; h < o_h; h++){

                double max_val = -DBL_MAX;
                int    max_id  = -1;

                // 이 출력 위치(oh, ow)에 대응하는 입력 window의 시작점
                int h_start = h * str - pad;
                int w_start = w * str - pad;

                for (int kh = 0; kh < prow; kh++) {
                    for (int kw = 0; kw < pcol; kw++) {
                        int ih = h_start + kh;
                        int iw = w_start + kw;

                        // 패딩 영역(이미지 밖)은 무시
                        if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w)
                            continue;

                        int in_idx = ch * (in_h * in_w) + ih * in_w + iw;
                        double v = input[in_idx];

                        if (v > max_val) {
                            max_val = v;
                            max_id  = in_idx;   // 입력 배열에서의 절대 인덱스
                        }
                    }
                }

                int out_idx = ch * (o_h * o_w) + h * o_w + w;
                pool->lpool[out_idx]  = max_val;
                pool->maxidx[out_idx] = max_id;
            }
        }
    }
    printf("PoolForward성공!!!!\n");
}
//==================================

// ============Flatten Forward============

void FlattenForward(POOL *pool, FLLAYER *fllayer)
{
    int in_c = pool->channel;
    int in_w = pool->out_pwidth;
    int in_h = pool->out_pheight;

    int nnodes = fllayer->nnodes;

    int expected = in_c * in_h * in_w;

    assert(nnodes == expected); // 앞뒤 레이어의 총 개수 같은지 확인

    for(int c = 0; c < in_c; c++){
        for(int h = 0; h < in_h; h++){
            for(int w = 0; w < in_w; w++){
                fllayer->outputs[(c * in_h * in_w) + (h * in_w) + w]
                = pool->lpool[(c * in_h * in_w) + (h * in_w) + w];
            }
        }
    }
    printf("FlattenForward성공!!!!\n");
}

// ==============FL2FC Forward===========
void FL2FCForward(FLLAYER *fllayer, FCLAYER *fclayer)
{
    int prev_nnodes = fllayer->nnodes;
    int nnodes = fclayer->nnodes;


    assert(prev_nnodes * nnodes == fclayer->nweights); // 실제 앞뒤 레이어 수로 계산한 weight 수와 할당한 weight공간 수 같은 지 확인 

    for(int n = 0; n < nnodes; n++){
        double sum = 0.0;
        for(int pn = 0; pn < prev_nnodes; pn++){
            double out = fllayer->outputs[pn] * fclayer->weights[(n * prev_nnodes) + pn];
            sum += out;
        }
        fclayer->outputs[n] = sum + fclayer->biases[n];
    }
    printf("FL2FCForward성공!!!!!\n");
}
//======================================

//================FC Forward=========
void FCForward(FCLAYER *prev_node, FCLAYER *curr_node)
{
    int prev_nnodes = prev_node->nnodes;
    int nnodes = curr_node->nnodes;

    printf("계산한 수%d\n", prev_nnodes * nnodes);
    printf("실제 수%d\n", curr_node->nweights);
    assert(prev_nnodes * nnodes == curr_node->nweights); // 실제 앞뒤 레이어 수로 계산한 weight 수와 할당한 weight공간 수 같은 지 확인 

    for(int n = 0; n < nnodes; n++){
        double sum = 0.0;
        for(int pn = 0; pn < prev_nnodes; pn++){
            double out = prev_node->z[pn] * curr_node->weights[(n * prev_nnodes) + pn];
            sum += out;
        }
        curr_node->outputs[n] = sum + curr_node->biases[n];
    }
    printf("FCForward 성공!!!!\n");
}

//===========Convolution Backward========
void PL2CVBackward(POOL *curr_layer, CONV *prev_layer)
{
    assert(curr_layer->channel == prev_layer->channel);
    for(int i = 0; i < (curr_layer->channel * curr_layer->out_pheight * curr_layer->out_pwidth); i++){
        int max_idx = curr_layer->maxidx[i];
        prev_layer->delta[max_idx] += curr_layer->delta[i];
    }
}


//==========pooling Backward============
void FL2PLBackward(FLLAYER *curr_layer, POOL *prev_layer)
{
    assert(curr_layer->nnodes == (prev_layer->channel * prev_layer->out_pheight * prev_layer->out_pwidth));
    for(int i = 0; i < curr_layer->nnodes; i++){
        prev_layer->delta[i] = curr_layer->delta[i];
    }
}

//=========FC2FLBackward================
void FC2FLBackward(FCLAYER *curr_layer, FLLAYER *prev_layer, double lr)
{
    double *delta = curr_layer->delta;
    double *input = prev_layer->outputs;
    int prev_nnode = prev_layer->nnodes;
    int curr_nnode = curr_layer->nnodes;

    // 이전 노드의 delta 업데이트
    for(int pn = 0; pn < prev_nnode; pn++){
        double sum = 0.0;
        for(int cn = 0; cn < curr_nnode; cn++){
            sum += curr_layer->delta[cn] * curr_layer->weights[cn * prev_nnode + pn];
        }
        prev_layer->delta[pn] = sum;
    }

    // dweight, dbias 계산 및 업데이트
    for(int cn = 0; cn < curr_nnode; cn++){
        for(int pn = 0; pn < prev_nnode; pn++){
            assert((curr_layer->nweights) == (curr_layer->nnodes * prev_layer->nnodes));
            curr_layer->dweights[cn * prev_nnode + pn] += curr_layer->delta[cn] * prev_layer->outputs[pn]; //dweight 누적
        }
        curr_layer->dbiases[cn] += curr_layer->delta[cn]; // dbias누적
    }
}

//===========FCBackward=================
void FCBackward(FCLAYER *curr_layer, FCLAYER *prev_layer, double lr)
{
    double *delta = curr_layer->delta;
    double *input = prev_layer->z;
    int prev_nnode = prev_layer->nnodes;
    int curr_nnode = curr_layer->nnodes;
	double* curr_z = curr_layer->z;

    // ReLU 미분 적용
    for (int cn = 0; cn < curr_nnode; cn++) {
        if (curr_layer->z[cn] <= 0.0) {
            curr_layer->delta[cn] = 0.0;
        }
    }


    // 이전 노드의 delta 업데이트
    for(int pn = 0; pn < prev_nnode; pn++){
        double sum = 0.0;
        for(int cn = 0; cn < curr_nnode; cn++){
            sum += curr_layer->delta[cn] * curr_layer->weights[cn * prev_nnode + pn];
        }
        prev_layer->delta[pn] = sum;
    }

    // dweight, dbias 계산 및 누적
    for(int cn = 0; cn < curr_nnode; cn++){
        for(int pn = 0; pn < prev_nnode; pn++){
            assert((curr_layer->nweights) == (curr_layer->nnodes * prev_layer->nnodes));
            curr_layer->dweights[cn * prev_nnode + pn] += curr_layer->delta[cn] * prev_layer->z[pn]; //dweight 누적
        }
        curr_layer->dbiases[cn] += curr_layer->delta[cn]; // dbias누적
    }
    
}

//=============kernel weight, bias 업데이트=========
void ConvBackward(CONV* conv, KERNEL* kernel)
{
    int C_in = kernel->in_c;       // 입력 채널 수
    int C_out = kernel->n_filter;   // 필터 수
    int k = kernel->k_size;
    int stride = conv->stride;
    int pad = conv->padding;

    int H_in = conv->in_height;
    int W_in = conv->in_width;

    int H_out = conv->out_cheight;
    int W_out = conv->out_cwidth;

    double* input = conv->input;      // [C_in][H_in][W_in]
    double* dY = conv->delta;      // [C_out][H_out][W_out]  (Pool에서 온 gradient)
    double* W = kernel->k_weights;
    double* dW = kernel->dweights;
    double* db = kernel->dbiases;

    // 1) Conv층 ReLU 미분 적용: conv->z <= 0 인 위치는 gradient 0
    for (int f = 0; f < C_out; f++) {
        for (int oy = 0; oy < H_out; oy++) {
            for (int ox = 0; ox < W_out; ox++) {
                int idx = f * (H_out * W_out) + oy * W_out + ox;
                if (conv->z[idx] <= 0.0) {
                    dY[idx] = 0.0;
                }
            }
        }
    }

    // 2) kernel weight, bias에 대한 gradient 누적
    for (int f = 0; f < C_out; f++) {
        for (int oy = 0; oy < H_out; oy++) {
            for (int ox = 0; ox < W_out; ox++) {

                int out_idx = f * (H_out * W_out) + oy * W_out + ox;
                double grad_out = dY[out_idx];  // dL/dY[f,oy,ox]

                if (grad_out == 0.0)
                    continue; // ReLU에서 죽은 뉴런은 스킵해도 됨

                // bias gradient: dL/db_f = Σ_{oy,ox} dY[f,oy,ox]
                db[f] += grad_out;

                // weight gradient:
                // dL/dW[f,c,ky,kx] += input[c,iy,ix] * dY[f,oy,ox]
                for (int c = 0; c < C_in; c++) {
                    for (int ky = 0; ky < k; ky++) {
                        for (int kx = 0; kx < k; kx++) {

                            int iy = oy * stride + ky - pad;
                            int ix = ox * stride + kx - pad;

                            if (iy < 0 || iy >= H_in || ix < 0 || ix >= W_in)
                                continue;

                            int in_idx = c * (H_in * W_in) + iy * W_in + ix;
                            int w_idx = f * (C_in * k * k)
                                + c * (k * k)
                                + ky * k + kx;

                            dW[w_idx] += input[in_idx] * grad_out;
                        }
                    }
                }
            }
        }
    }
}

// ===========weight, bias 업데이트=========
void UpdateFCWeightsBiases(FCLAYER *curr_layer, FCLAYER *prev_layer, double lr, unsigned int batch_size)
{
    int prev_nnode = prev_layer->nnodes;
    int curr_nnode = curr_layer->nnodes;
    // weight, bias 업데이트
    for(int cn = 0; cn < curr_nnode; cn++){
        for(int pn = 0; pn < prev_nnode; pn++){
            curr_layer->weights[cn * prev_nnode + pn] -= lr * (curr_layer->dweights[cn * prev_nnode + pn] / batch_size); //weight 업데이트
        }
        curr_layer->biases[cn] -= lr * (curr_layer->dbiases[cn] / batch_size); // bias업데이트
    }
}

void UpdateKernelWeightsBiases(KERNEL *kernel_layer, double lr, unsigned int batch_size)
{
    int nweights = kernel_layer->nweights;
    int nbiases = kernel_layer->nbiases;
    // weight, bias 업데이트
    for(int w = 0; w < nweights; w++){
        kernel_layer->k_weights[w] -= lr * (kernel_layer->dweights[w] / batch_size) ; //weight 업데이트
    }
    for(int b = 0; b < nbiases; b++){
        kernel_layer->k_biases[b] -= lr * (kernel_layer->dbiases[b] / batch_size); // bias업데이트
    }
}

// ======활성화 함수를 위한 ReLU함수======
void ReLU(const double *input, double *output, int n_in)
{
    for (int i = 0; i < n_in; i++) {
        output[i] = (input[i] > 0.0 ? input[i] : 0.0);
    }
    printf("ReLU성공!!!!\n");
}

// =====마지막 최종 확률을 구하기 위한 SoftMax함수========

void SoftMax(const double *input, double *output, int n_in)
{

    double max = input[0]; // overflow방지를 위해 input에 max값 빼줘야 함
    for (int i = 1; i < n_in; i++) {
        if (input[i] > max) max = input[i];
    }

    // exp(x_i - max) 계산하면서 합 구하기
    double sum = 0.0;
    for (int i = 0; i < n_in; i++) {
        output[i] = exp(input[i] - max);
        sum += output[i];
    }

    for (int i = 0; i < n_in; i++) {
        output[i] /= sum;
    }

}
// =====배치별 평균 Binary Cross Entropy손실 함수======
// 호출 하기 전 y_true값을 double 형인지 확인해야함

double *BCE(const double *y_true, const double *y_pred, int nnodes, int batch)
{
    const double eps = 1e-7;   // 0, 1에 너무 가까울 때 log 커지는 상황 방지
    

    double *p_loss = calloc(nnodes, sizeof(double));

    for(int node = 0; node < nnodes; node++){
        double loss_sum = 0.0;
        for (int bat = 0; bat < batch; bat++) {
        double p = y_pred[bat * nnodes + node];
        double t = y_true[bat * nnodes + node];
        // 범위 클리핑
        if (p < eps) p = eps;
        else if (p > 1.0 - eps) p = 1.0 - eps;

        loss_sum += -( t * log(p) +
                       (1.0 - t) * log(1.0 - p) );
        }
        p_loss[node] = (loss_sum / batch);
    }
    // 배치 평균 loss
    return p_loss;
}

// =========마지막 층에서의 delta값 계산========
void FinalDelta(const double *y_true, const double *y_pred, double *delta, int nnodes, int batch)
{
    const double eps = 1e-7;   // 0, 1에 너무 가까울 때 log 커지는 상황 방지
   
    for(int node = 0; node < nnodes; node++){
        double p = y_pred[node];
        double t = y_true[node];
        // 범위 클리핑
        if (p < eps) p = eps;
        else if (p > 1.0 - eps) p = 1.0 - eps;
        delta[node] = (p - t); // del binary_cross_entropy x del softmax 함수
    }
}

//===================dweight, dbias 초기화=================
void InitDWeightBias(double* dweights, double* dbiases, int nweights, int nbiases)
{
    for(int i = 0; i < nweights; i++){
        dweights[i] = 0.0;
    }
    for(int j = 0; j < nbiases; j++){
        dbiases[j] = 0.0;
    }
}

//=====================delta 초기화=======================
void InitDelta(double* delta, int nnodes)
{
    for (int i = 0; i < nnodes; i++) {
        delta[i] = 0.0;
    }
}

