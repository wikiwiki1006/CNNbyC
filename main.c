#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <math.h>

#include "fread.h"
#include "layer.h"
#include "calc.h"

// #define DEBUG

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
    // 전체 이미지 수 : 60000개
	unsigned int train_size = (unsigned int)(mnist.num_imgs * 0.8);
	unsigned int test_size = (unsigned int)(mnist.num_imgs * 0.2);

    /*
    모델 생성
	1)  input 1x28x28
	2)  kernel conv층 32개, 커널크기 3x3, 스트라이드 1, 패딩 1
	3)  convolution 층 -> output 32x28x28
	4)  ReLU 활성화 함수
	5)  max pooling 층, 필터 크기 2x2, 스트라이드 2 -> output 32x14x14
	6)  flatten 층 -> output 32 x 14 x 14 = 6272
	7)  fully connected 층 128노드, ReLU 활성화 함수
    8)  fully connected 층 64노드, ReLU 활성화 함수
	9)  fully connected 층 10노드, softmax 활성화 함수
*/
//====================모델 층 생성====================
	KERNEL *kernel1 = AddKernelLayer(1, 28, 28, 32, 3);
	CONV *conv1 = AddConv(32, 28, 28, 3, 1, 1);
	POOL *pool1 = AddPool(32, 28, 28, 2, 2, 0, 2);
	FLLAYER *flat = AddFlattenLayer(32, 14, 14);
    FCLAYER *fc1 = AddFCLayer(128, 6272);
	FCLAYER *fc2 = AddFCLayer(64, 128);
	FCLAYER *final = AddFCLayer(10, 64);
//==============================================

    //학습률, 에포크, 배치수 설정
    double lr = 0.05;
    unsigned int epoch = 5;
    unsigned int batch_size = 100;
    unsigned int batch_;
    unsigned int iter;
    unsigned int final_nnodes = final->nnodes;
    unsigned int rem = train_size % batch_size;

    rem == 0 ? (iter = train_size / batch_size) : (iter = train_size / batch_size + 1);
    for(unsigned int ep = 0; ep < epoch; ep++)
    {
        double epoch_loss = 0.0;
        unsigned int processed = 0;
        for(unsigned int it = 0; it < iter; it++)
        { 
            ((it == (iter - 1) && rem != 0) ? (batch_ = rem) : (batch_ = batch_size)); // 나머지 때문에 마지막 배치수가 지정한 배치 수와 다를 때
			// dweight, dbias 0으로 초기화
            InitDWeightBias(kernel1->dweights, kernel1->dbiases, kernel1->nweights, kernel1->nbiases);
            InitDWeightBias(fc1->dweights, fc1->dbiases, fc1->nweights, fc1->nbiases);
            InitDWeightBias(fc2->dweights, fc2->dbiases, fc2->nweights, fc2->nbiases);
            InitDWeightBias(final->dweights, final->dbiases, final->nweights, final->nbiases);
            double batch_loss = 0.0; 

            for(unsigned int bat = 0; bat < batch_; bat++)
            {
                InitDelta(fc2->delta, fc2->nnodes);
				InitDelta(fc1->delta, fc1->nnodes);
				InitDelta(final->delta, final->nnodes);
				InitDelta(conv1->delta, conv1->channel * conv1->out_cheight * conv1->out_cwidth);

                double* y_pred = (double*)calloc(final_nnodes, sizeof(double));
                double* y_true = (double*)calloc(final_nnodes, sizeof(double));

                unsigned int img_idx = (it * batch_size + bat);
                unsigned int img_size = mnist.cols * mnist.rows;
                int label = mnist.labels[img_idx];
                assert(label >= 0 && label < (int)final_nnodes);
                y_true[label] = 1.0;
                ConvForward(conv1, mnist.norm_images + (img_idx * img_size),  kernel1);
                ReLU(conv1->outputs, conv1->z, conv1->channel * conv1->out_cheight * conv1->out_cwidth);
                PoolForward(pool1, conv1);
                FlattenForward(pool1, flat);
                FL2FCForward(flat, fc1);
                ReLU(fc1->outputs, fc1->z, fc1->nnodes);
                FCForward(fc1, fc2);
                ReLU(fc2->outputs, fc2->z, fc2->nnodes);
                FCForward(fc2, final);
                SoftMax(final->outputs, final->z, final->nnodes);
                // 배치 수 만큼 y_true_label, y_pred_label저장
                for(unsigned int nnode = 0; nnode < final_nnodes; nnode++){
                    y_pred[nnode] = final->z[nnode];
                }
                assert(final_nnodes == final->nnodes);

                const double eps = 1e-7;
                double p_true = y_pred[label];
                if (p_true < eps) p_true = eps;
                batch_loss += -log(p_true);  // binary-cross-entropy 손실 값 누적
                
                // dweight, dbias 누적
                FinalDelta(y_true, y_pred, final->delta, final_nnodes, batch_);
                FCBackward(final, fc2, lr);
                FCBackward(fc2, fc1, lr);
                FC2FLBackward(fc1, flat, lr);
                FL2PLBackward(flat, pool1);
                PL2CVBackward(pool1, conv1);
                ConvBackward(conv1, kernel1);
                free(y_pred);
                free(y_true);

                processed++;
                printf("\r[Epoch %u/%u] %u개 중 %u개 완료 ", ep + 1, epoch, train_size, processed);
                fflush(stdout); 
            }

			// weight, bias 업데이트
			UpdateFCWeightsBiases(final, fc2, lr, batch_);
            UpdateFCWeightsBiases(fc2, fc1, lr, batch_);
            UpdateFCWeightsBiases(fc1, flat, lr, batch_);
            UpdateKernelWeightsBiases(kernel1, lr, batch_);

            // 배치 당 손실 평균
            double loss_per_batch = batch_loss / (double)batch_;
            if((it != 0) && (it % 50 == 49)) printf("iter %u batch_loss = %.6f\n", it + 1, loss_per_batch); // 50 iteration 마다 평균 손실 값 출력

            epoch_loss += loss_per_batch;
 

        }

        printf("%d 번째 에포크 완료!\n", ep + 1);
        printf("epoch loss = %f\n\n", epoch_loss / iter);

    }




    // ========================추론==========================

    int acc = 0;
    printf("===============추론 시작=================");

    for(int k = 0; k < test_size; k++){
        ConvForward(conv1, mnist.norm_images + ((train_size + k) * 28*28),  kernel1);
        ReLU(conv1->outputs, conv1->z, conv1->channel * conv1->out_cheight * conv1->out_cwidth);
        PoolForward(pool1, conv1);
        FlattenForward(pool1, flat);
        FL2FCForward(flat, fc1);
        ReLU(fc1->outputs, fc1->z, fc1->nnodes);
        FCForward(fc1, fc2);
        ReLU(fc2->outputs, fc2->z, fc2->nnodes);
        FCForward(fc2, final);
        SoftMax(final->outputs, final->z, final->nnodes);

        int max_idx = 0;
        double max = 0.0;

        for(int i = 0; i < final->nnodes; i++){
            if(final->z[i] >= max) {
                max_idx = i;
                max = final->z[i];
            }
        }
        
        if(max_idx == mnist.labels[train_size + k]) acc += 1;

        if((test_size - k) < 5){ // 마지막 샘플 5개 시각화
            for(int i = 0; i < final->nnodes; i++){
                printf("%d 확률 : %.5f\n", i, final->z[i]);
                if(final->z[i] >= max){
                    max_idx = i;
                    max = final->z[i];
                }
            }
            
            printf("\n예측 값 : %d\n", max_idx);

            printf("실제 값 : %d\n", mnist.labels[train_size + k]);
            for(int i = 0; i < 28; i++){
                for(int j = 0; j< 28; j++){
                    double pixel = mnist.norm_images[((train_size + k) *28*28) + (28*i + j)];
                    putchar(pixel > 0.3 ? '#' : ' ');
                }
                printf("\n");
            }
            printf("\n==========================\n");
        }

    }
    double accuracy = (double) acc / (double) test_size;
    printf("테스트 셋 정확도 : %.5f%%\n", accuracy * 100);

    printf("\n");

    FreeMNIST(&mnist);
    free(mnist.norm_images);

    return 0;
}
   