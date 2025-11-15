#include <stdio.h>
#include <stdlib.h>
#include "fread.h"

int main(void) {
    MNIST mnist;
    if (Freader(&mnist) != 0) {
        return 1;
    }
    
    float *norm = malloc(sizeof(float) * mnist.num_imgs * mnist.rows * mnist.cols);
    normalize_imgs(&mnist, norm); // 이미지 픽셀값 [0, 1]정규화

    // 예시: 첫 3개 이미지의 라벨과 픽셀 출력
    for (int n = 0; n < 3; n++) {
        printf("[%d] 라벨 = %d\n", n, mnist.labels[n]);
        for (int i = 0; i < mnist.rows; i++) {
            for (int j = 0; j < mnist.cols; j++) {
                printf("%.5f ", norm[i * mnist.cols + j]);
            }
            putchar('\n');
        }
        putchar('\n');
    }

    free(mnist.images);
    free(mnist.labels);
    return 0;
}