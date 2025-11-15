cmake_minimum_required(VERSION 3.10)
project(MNIST_Reader C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 헤더 파일 포함
include_directories(include)

# 🔹 실행할 소스 지정 (원하는 파일만 추가)
add_executable(mnist_reader
    main.c
    src/layer.c
    src/fread.c
)

# 컴파일 옵션
target_compile_options(mnist_reader PRIVATE -Wall -O2)

# 데이터 폴더를 build 폴더로 복사 (옵션)
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})