cmake_minimum_required(VERSION 3.10)
project(MNIST_Reader C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(cnn_mnist
    main.c
    src/layer.c
    src/fread.c
    src/calc.c
)

target_compile_options(cnn_mnist PRIVATE -Wall)

file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

add_compile_options("/utf-8")
