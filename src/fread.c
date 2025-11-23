#include <stdio.h>
#include <stdlib.h>
#include "fread.h"

#define PRINT 1 // mnist 3개 그림 출력

/*
MNIST byte 설명 링크
https://medium.com/the-owl/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
*/


unsigned int read_uint(FILE *fp) {
    unsigned char data[4];
    fread(data, sizeof(data), 1, fp);
    
    // mnist의 ubyte는 Big-endian형식 -> macOS, Window에 맞게 Littel-endian방식으로 변환
    return (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
}

int Freader(MNIST *data) {
    FILE *fp_imgs = fopen("./data/train_images.idx3-ubyte", "rb");
    FILE *fp_lbls = fopen("./data/train_labels.idx1-ubyte", "rb");

    if (!fp_imgs || !fp_lbls) {
        printf("파일이 없습니다.\n");
        return 1;
    }

    // mnist 형식에 맞게 header읽기 
    read_uint(fp_imgs);
    data->num_imgs = read_uint(fp_imgs);
    data->rows = read_uint(fp_imgs);
    data->cols = read_uint(fp_imgs);

    read_uint(fp_lbls); // magic number skip
    data->num_labels = read_uint(fp_lbls);

    size_t img_size = data->rows * data->cols;

    data->images = (unsigned char *)malloc(data->num_imgs * img_size);
    data->norm_images = (double *)calloc(data->num_imgs * img_size, sizeof(double));
    data->labels = (unsigned char *)malloc(data->num_imgs);
    fread(data->images, img_size, data->num_imgs, fp_imgs);
    fread(data->labels, 1, data->num_labels, fp_lbls);


    #if PRINT
    for (int k = 0; k < 3; k++) {
        printf("label of %dth image: %d\n", k + 1, data->labels[k]);

        // 각 이미지의 시작 인덱스 계산
        unsigned char *img_ptr = data->images + k * (data->rows * data->cols);

        // --- 이미지 출력 (28x28) ---
        for (int i = 0; i < data->rows; i++) {
            for (int j = 0; j < data->cols; j++) {
                unsigned char pixel = img_ptr[i * data->cols + j];
                putchar(pixel > 128 ? '#' : ' ');
            }
            putchar('\n');
        }
        putchar('\n');
    }
    #endif

        fclose(fp_imgs);
        fclose(fp_lbls);
        return 0;
    }

void FreeMNIST(MNIST *data)
    {
        if (data->images != NULL) {
            free(data->images);
            data->images = NULL;
        }
        if (data->labels != NULL) {
            free(data->labels);
            data->labels = NULL;
        }
    }
    
void normalize_imgs(MNIST *data) //이미지 픽셀 값 [0, 1] 정규화  
{
    unsigned int total;
    total = data->num_imgs * data->rows * data->cols;
    for(int i = 0; i < total; i++){
        data->norm_images[i] = data->images[i] / 255.0;
    }
}

