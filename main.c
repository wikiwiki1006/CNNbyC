#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "fread.h"
#include "layer.h"

// #define DEBUG

int main(void) {
    MNIST mnist;
    if (Freader(&mnist) != 0) {
        return 1;
    }

    float *norm_imgs = malloc(sizeof(float) * mnist.num_imgs * mnist.rows * mnist.cols);
    normalize_imgs(&mnist, norm_imgs); // 이미지 픽셀값 [0, 1]정규화

    srand((unsigned int)time(NULL)); // 시간 기반 seed 초기화

    KERNEL kernel;

    KernelAddLayer(&kernel, 3, 3, 3);
    printf("\n[첫 번째 커널 일부 값 미리보기]\n");
    for (int i = 0; i < 10; i++) {
        printf("%.5f ", kernel.kernel[i]);
        
    }
    printf("\n");
    #ifdef DEBUG
    // 예시: 첫 3개 이미지의 라벨과 픽셀 출력
    for (int n = 0; n < 3; n++) {
        printf("[%d] 라벨 = %d\n", n, mnist.labels[n]);
        for (int i = 0; i < mnist.rows; i++) {
            for (int j = 0; j < mnist.cols; j++) {
                printf("%.2f ", norm_imgs[i * mnist.cols + j]);
            }
            putchar('\n');
        }
        putchar('\n');
    }
    #endif

    FreeMNIST(&mnist);
    FreeKernel(&kernel);
    free(norm_imgs);

    return 0;
}