# C언어를 사용한 CNN모델 생성 및 MNIST 학습

## 데이터 입력
MNIST   
- 데이터 구조 : Train data 60000개, Test data 10000개, .idx파일

- 이미지 구조 : 28 X 28, 값 0 ~ 255

- MNIST데이터 .idx 파일 입력

## 데이터 처리
Normalization : [1, 255] -> [0, 1] 


## FC(fully connected layer) 설계

## 손실함수
### Binary Cross Entropy
$$
\text{Loss} = -\frac{1}{{\text{outputsize}}}
\sum_{i=1}^{\text{outputsize}}
\left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right]
$$
## 초기화 함수
### he Initalization
$$
w \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan-in}}}\right)
$$

## kernel
Size : 3 x 3    
초기화 : He Initialization (Kaiming Initialization) 

## Convolution 층
size :  
row = 
$$  
\frac{\text{in\_height} + 2 \cdot \text{padding} - \text{k\_size}}{\text{stride}} + 1   
$$
col =   
$$ \frac{\text{in\_width} + 2 \cdot \text{padding} - \text{k\_size}}{\text{stride}} + 1 $$

## Pooling 층

## 활성화 함수
RELU : max(0, x)

## Optimizer
Not used

## 파일 구조
```
Cproject
├─ CMakeLists.txt
├─ README.md
├─ data
│  ├─ test_images.idx3-ubyte
│  ├─ test_labels.idx1-ubyte
│  ├─ train_images.idx3-ubyte
│  └─ train_labels.idx1-ubyte
├─ include
│  ├─ calc.h
│  ├─ fread.h
│  └─ layer.h
├─ src
│  ├─ calc.c
│  ├─ fread.c
│  └─ layer.c
└─  main.c


```