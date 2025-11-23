#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "fread.h"
#include "layer.h"
#include "calc.h"

#define DEBUG
#define SOFTMAX = 1
#define RELU = 0

int main(void) {
    /* Mnist 데이터 읽어오기*/
    MNIST mnist;
    if (Freader(&mnist) != 0) return 1;

    normalize_imgs(&mnist); // 이미지 픽셀값 [0, 1]정규화

    srand((unsigned int)time(NULL)); // 시간 기반 seed 초기화

    // 데이터 랜덤 셔플
    for (unsigned int i = 0; i < mnist.num_imgs; i++) {
        unsigned int rand_idx = rand() % mnist.num_imgs;
        // 이미지 교환
        for (unsigned int j = 0; j < mnist.rows * mnist.cols; j++) {
            double temp = mnist.norm_images[i * mnist.rows * mnist.cols + j];
            mnist.norm_images[i * mnist.rows * mnist.cols + j] = mnist.norm_images[rand_idx * mnist.rows * mnist.cols + j];
            mnist.norm_images[rand_idx * mnist.rows * mnist.cols + j] = temp;
        }
        // 라벨 교환
        unsigned char temp_label = mnist.labels[i];
        mnist.labels[i] = mnist.labels[rand_idx];
        mnist.labels[rand_idx] = temp_label;
	}

	// 훈련, 검증, 테스트 데이터셋 분할
	// unsigned int train_size = (unsigned int)(mnist.num_imgs * 0.8);
	// unsigned int val_size = (unsigned int)(mnist.num_imgs * 0.1);
	// unsigned int test_size = mnist.num_imgs - train_size - val_size;
    unsigned int train_size = 5;
    
    /*
    모델 생성
	1)  input 1x28x28
	2)  kernel conv층 32개, 커널크기 3x3, 스트라이드 1, 패딩 0
	3)  convolution 층 -> output 32x26x26
	4)  ReLU 활성화 함수
	5)  max pooling 층, 필터 크기 2x2, 스트라이드 2 -> output 32x13x13
	6)  flatten 층 -> output 5408
	7)  fully connected 층 128노드, ReLU 활성화 함수
	8)  fully connected 층 10노드, softmax 활성화 함수
*/
//================모델 층 생성====================
	KERNEL *kernel1 = AddKernelLayer(1, 28, 28, 32, 3);
	CONV *conv1 = AddConv(32, 28, 28, 3, 1, 1);
	POOL *pool1 = AddPool(32, 28, 28, 2, 2, 0, 2);
	FLLAYER *flat = AddFlattenLayer(32, 14, 14);
    FCLAYER *fc1 = AddFCLayer(128, 6272);
	FCLAYER *fc2 = AddFCLayer(64, 128);
	FCLAYER *final = AddFCLayer(10, 64);
//==============================================

    //학습률, 에포크, 배치수
    double lr = 0.001;
    unsigned int epoch = 1;
    unsigned int batch_size = 10;
    unsigned int batch_;
    unsigned int iter;
    unsigned int final_nnodes = final->nnodes;
    unsigned int pos_img; // 입력 이미지 위치
    // iteration 계산
    unsigned int rem = train_size % batch_size;
    rem == 0 ? (iter = train_size / batch_size) : (iter = train_size / batch_size + 1);
    for(unsigned int ep = 0; ep < epoch; ep++)
    {
        for(unsigned int it = 0; it < iter; it++)
        { 
            ((it == (iter - 1)) ? (batch_ = rem) : (batch_ = batch_size)); // 나머지 때문에 마지막 배치수가 지정한 배치 수와 다를 때
            double *y_pred = (double *)calloc(final_nnodes * batch_, sizeof(double));
            double *y_true = (double *)calloc(final_nnodes * batch_, sizeof(double));

            for(unsigned int bat = 0; bat < batch_; bat++)
            {
                unsigned int img_idx = (it * batch_size + bat);
                unsigned int img_size = mnist.cols * mnist.rows;
                int label = mnist.labels[img_idx];
                printf("%d\n", label);
                assert(label >= 0 && label < (int)final_nnodes);
                y_true[final_nnodes * bat + label] = 1.0;
                ConvForward(conv1, mnist.norm_images + (img_idx * img_size),  kernel1);
                conv1->z = ReLU(conv1->outputs, conv1->channel * conv1->out_cheight * conv1->out_cwidth);
                PoolForward(pool1, conv1);
                FlattenForward(pool1, flat);
                FL2FCForward(flat, fc1);
                fc1->z = ReLU(fc1->outputs, fc1->nnodes);
                FCForward(fc1, fc2);
                fc2->z = ReLU(fc2->outputs, fc2->nnodes);
                FCForward(fc2, final);
                final->z = SoftMax(final->outputs, final->nnodes);
                // 배치 수 만큼 y_true_label, y_pred_label저장
                for(unsigned int nnode = 0; nnode < final_nnodes; nnode++){
                    y_pred[bat * final_nnodes + nnode] = final->z[nnode];
                }
                assert(final_nnodes == final->nnodes);
                // final->errors = BCE(y_true, y_pred, final_nnodes, batch_);
                final->delta = FinalDelta(y_true, y_pred, final_nnodes, batch_);
            }
            free(y_pred);
            free(y_true);
            // 손실 평균
            printf("손실값\n");
            for(int k = 0; k < final_nnodes; k++){
                printf("%.3f  ", final->delta[k]);
            }
            FCBackward(final, fc2, lr);
            FCBackward(fc2, fc1, lr);
            FC2FLBackward(fc1, flat, lr);
            FL2PLBackward(flat, pool1);
            // grad계산
            // 가중치 업뎃
            // error 초기화
        }
        // error 초기화
    }



    // double *outputs = conv1->outputs;
    // printf("\n전");
    // for(int i = 0; i < 784; i++){
    //     printf("%.3f ", outputs[i]);
    // }
    printf("\n후\n");
    for(int i = 0; i < 28; i++){
        for(int j = 0; j< 28; j++){
            printf("%.1f ", conv1->z[28*i + j]);
        }
        printf("\n");
    }
    printf("\n\n");
    for(int i = 0; i < 14; i++){
        for(int j = 0; j< 14; j++){
            printf("%.1f ", pool1->lpool[14*i + j]);
        }
        printf("\n");
    }
    printf("\n");
    for(int i = 0; i < final->nnodes; i++){
        printf("%.4f\n", final->z[i]);
    }
	/* 모델 학습
	* psuedo code
    * 
    * //순전파
	* conv1_forward(norm_imgs, kernel_layer, conv_layer); // conv_layer->outputs에 결과 저장
	* ReLU_forward(conv_layer);
	* pool1_forward(conv_layer, pool_layer);
	* flatten_forward(pool_layer, flatten_layer);
	* fc1_forward(flatten_layer, fc1_layer);
    * ReLU_forward(flatten_layer);
	* fc2_forward(fc1_layer, fc2_layer);
	* output = SoftMax(fc2_layer->outputs);
    * 
    * //역전파
    * 
	* 1) 손실함수 : binary cross-entropy
	* 2) 최적화 기법 : Adam
	* 3) 배치 크기 : 64
	* 4) 에포크 : 10
	* 5) 학습률 : 0.001
	* 6) 검증 데이터셋으로 매 에포크마다 모델 평가
    */
    

	/*
    테스트 데이터셋으로 모델 평가
	1) 정확도(accuracy) 출력
    */


    printf("\n");
    #ifdef DEBUG
    // 예시: 첫 3개 이미지의 라벨과 픽셀 출력
    for (int n = 0; n < 3; n++) {
        printf("[%d] LABEL = %d\n", n, mnist.labels[n]);
        for (unsigned int i = 0; i < mnist.rows; i++) {
            for (unsigned int j = 0; j < mnist.cols; j++) {
                printf("%.2f ", mnist.norm_images[i * mnist.cols + j]);
            }
            putchar('\n');
        }
        putchar('\n');
    }
    #endif

    FreeMNIST(&mnist);
    free(mnist.norm_images);

    return 0;
    }