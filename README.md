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
## kernel
Size : 3 x 3    
초기화 : He Initialization (Kaiming Initialization)
## Convolution 층

## Pooling 층

## 활성화 함수
RELU : max(0, x)

## Optimizer

## 순전파

## 역전파

## SGD
